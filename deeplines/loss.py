import numpy as np
import torch
from torch import nn

from . import utils
from .line import Line


class DeepLineLoss(nn.Module):
    def __init__(self, image_size: tuple[int, int], n_columns: int, anchors_per_column: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_columns = n_columns
        self.anchors_per_column = anchors_per_column
        self.bce = torch.nn.BCELoss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, gt: list[list[Line]]) -> dict[str, torch.Tensor]:
        lines_batch = utils.get_lines_from_output(pred, self.image_size[0], self.image_size[1], threshold=0)

        p0_x, p0_y, p1_x, p1_y = self.get_points_from_pred(pred)

        distances_batch = []
        best_match_batch = []

        for gt_image, lines in zip(gt, lines_batch):
            assert len(lines) == self.n_columns * self.anchors_per_column
            best_match_image = []
            distances_image = []

            gt_by_column = self.get_lines_grouped_by_column(gt_image)

            for i in range(self.n_columns):
                if len(gt_by_column[i]) == 0:
                    best_match_image += [None] * self.anchors_per_column
                    distances_image += [float('inf')] * self.anchors_per_column
                    continue
                column_lines = lines[self.anchors_per_column * i:self.anchors_per_column * (i + 1)]
                distances = utils.get_distance_between_lines(gt_by_column[i], column_lines)

                best_match_column = []

                for n, _ in enumerate(column_lines):
                    best_match_idx = np.argmin(distances[:, n])
                    best_match_column.append(gt_by_column[i][best_match_idx])
                    distances_image.append(distances[best_match_idx, n])
                best_match_image += best_match_column

            best_match_batch.append(best_match_image)
            distances_batch.append(distances_image)

        best_p0_x = torch.zeros_like(p0_x)
        best_p0_y = torch.zeros_like(p0_y)
        best_p1_x = torch.zeros_like(p1_x)
        best_p1_y = torch.zeros_like(p1_y)
        for img in range(len(p0_x)):
            for column in range(len(p0_x[img])):
                for line in range(len(p0_x[img, column])):

                    best_match = best_match_batch[img][column * self.anchors_per_column + line]

                    if best_match is None:
                        continue

                    match_p0 = best_match.p0()
                    match_p1 = best_match.p1()

                    best_p0_x[img, column, line] = match_p0[0]
                    best_p0_y[img, column, line] = match_p0[1]
                    best_p1_x[img, column, line] = match_p1[0]
                    best_p1_y[img, column, line] = match_p1[1]

        hausdorff_distance, _ = torch.max(
            torch.stack((
                torch.min(
                    self.get_euclidean_distance(p0_x, p0_y, best_p0_x, best_p0_y),
                    self.get_euclidean_distance(p0_x, p0_y, best_p1_x, best_p1_y),
                    ),
                torch.min(
                    self.get_euclidean_distance(p1_x, p1_y, best_p0_x, best_p0_y),
                    self.get_euclidean_distance(p1_x, p1_y, best_p1_x, best_p1_y),
                    ),
                torch.min(
                    self.get_euclidean_distance(p0_x, p0_y, best_p0_x, best_p0_y),
                    self.get_euclidean_distance(p1_x, p1_y, best_p0_x, best_p0_y),
                    ),
                torch.min(
                    self.get_euclidean_distance(p0_x, p0_y, best_p1_x, best_p1_y),
                    self.get_euclidean_distance(p1_x, p1_y, best_p1_x, best_p1_y),
                    ),
                ), dim=0), dim=0
            )

        objectness = self.get_objectness_from_gt(gt, distances_batch)
        objects_mask = self.get_objectness_mask(gt)
        objectness = objectness.to(pred.device)
        objects_mask = objects_mask.to(pred.device)

        loss_objectness = 0.2 * self.mse(objects_mask * pred[:, :, :, 0:1], objects_mask * objectness)
        loss_no_objectness = 0.1 * self.mse((1 - objects_mask) * pred[:, :, :, 0:1], (1 - objects_mask) * objectness)
        # loss_objectness = 0.7 * self.bce(pred[:, :, :, 0:1], objects_mask)
        # loss_no_objectness = 0.15 * self.mse((1 - objects_mask) * pred[:, :, :, 0:1], (1 - objects_mask) * objectness)
        loss_distance = 0.7 * torch.mean(objects_mask.squeeze(-1) * hausdorff_distance)

        loss = {
            'distance': loss_distance,
            'objectness': loss_objectness,
            'no_objectness': loss_no_objectness,
        }

        return loss

    def get_objectness_from_gt(self, gt: list[list[Line]], distances_batch: list[list[float]]) -> torch.Tensor:
        objectness = torch.zeros((len(gt), self.n_columns, self.anchors_per_column, 1))

        for i in range(len(gt)):
            for j in range(self.n_columns):
                for k in range(self.anchors_per_column):
                    objectness[i, j, k, 0] = self.distance_to_confidence(
                        distances_batch[i][j * self.anchors_per_column + k],
                    )
        return objectness

    def distance_to_confidence(self, distance: float) -> float:
        return 1 - (distance / 40) if distance < 40 else 0

    def get_points_from_pred(self, pred: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cx = pred[:, :, :, 1] * self.image_size[0]
        cy = pred[:, :, :, 2] * self.image_size[1]
        angle = pred[:, :, :, 3] * np.pi
        length = pred[:, :, :, 4] * self.image_size[0]

        p0_x = cx - (length * torch.cos(angle) / 2)
        p0_y = cy - (length * torch.sin(angle) / 2)

        p1_x = cx + (length * torch.cos(angle) / 2)
        p1_y = cy + (length * torch.sin(angle) / 2)

        return p0_x, p0_y, p1_x, p1_y

    @staticmethod
    def get_euclidean_distance(p0_x: torch.Tensor, p0_y: torch.Tensor, p1_x: torch.Tensor, p1_y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((p0_x - p1_x) ** 2 + (p0_y - p1_y) ** 2)

    def get_objectness_mask(self, gt: list[list[Line]]) -> torch.Tensor:
        objectness = torch.zeros((len(gt), self.n_columns, self.anchors_per_column, 1))

        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = int(gt[i][j].cx * self.n_columns // self.image_size[1])
                if x == self.n_columns:
                    x = self.n_columns - 1
                objectness[i, x, :, 0] = 1
        return objectness

    def get_lines_grouped_by_column(self, gt_image: list[Line]):
        out = [[] for _ in range(self.n_columns)]
        for i in gt_image:
            x = int(i.cx * self.n_columns // self.image_size[0])
            if x == self.n_columns:
                x = self.n_columns - 1
            out[x].append(i)
        return out
