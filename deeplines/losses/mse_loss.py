import numpy as np
import torch
from torch import nn

from .. import utils
from ..line import Line


class MSELoss(nn.Module):
    def __init__(self, image_size: tuple[int, int], n_columns: int, anchors_per_column: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.n_columns = n_columns
        self.anchors_per_column = anchors_per_column
        self.bce = torch.nn.BCELoss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred: torch.Tensor, gt: list[list[Line]]) -> dict[str, torch.Tensor]:
        lines_batch = utils.get_lines_from_output(pred, self.image_size[0], self.image_size[1], threshold=0)
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

        objectness = self.get_objectness_from_gt(gt, distances_batch)
        objects_mask = self.get_objectness_mask(gt)
        objectness = objectness.to(pred.device)
        objects_mask = objects_mask.to(pred.device)

        regression = self.get_regression_from_best_match(lines_batch, best_match_batch)
        regression = regression.to(pred.device)

        loss_objectness = 0.1 * self.bce(objects_mask * pred[:, :, :, 0:1], objects_mask * objectness)
        loss_no_objectness = 0.1 * self.mse((1 - objects_mask) * pred[:, :, :, 0:1], (1 - objects_mask) * objectness)
        loss_center = 0.25 * self.mse(objects_mask * pred[:, :, :, 1:3], objects_mask * regression[:, :, :, :2])
        loss_angle = 0.25 * self.mse(objects_mask * pred[:, :, :, 3:4], objects_mask * regression[:, :, :, 2:3])
        loss_length = 0.3 * self.mse(objects_mask * pred[:, :, :, 4:], objects_mask * regression[:, :, :, 3:])
        assert regression[:, :, :, 3:].min() >= 0
        assert regression[:, :, :, 3:].max() <= 1

        loss = {
            'center': loss_center,
            'angle': loss_angle,
            'length': loss_length,
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
        return 1 - (distance / 20) if distance < 20 else 0

    def get_regression_from_best_match(self, lines_batch: list[list[Line]], best_match_batch: list[list[Line]]) -> torch.Tensor:
        regression = torch.zeros((len(lines_batch), self.n_columns, self.anchors_per_column, 4))

        for img_idx in range(len(lines_batch)):
            for i in range(self.n_columns):
                for j in range(self.anchors_per_column):
                    if best_match_batch[img_idx][i * self.anchors_per_column + j] is None:
                        continue
                    cx = best_match_batch[img_idx][i * self.anchors_per_column + j].cx
                    cy = best_match_batch[img_idx][i * self.anchors_per_column + j].cy
                    angle = best_match_batch[img_idx][i * self.anchors_per_column + j].angle
                    length = best_match_batch[img_idx][i * self.anchors_per_column + j].length

                    regression[img_idx, i, j, 0] = cx / self.image_size[0]
                    regression[img_idx, i, j, 1] = cy / self.image_size[1]
                    regression[img_idx, i, j, 2] = angle / np.pi
                    regression[img_idx, i, j, 3] = length / self.image_size[0]
                    if regression[img_idx, i, j, :].max() > 1:
                        print(regression[img_idx, i, :])
                        print(cx, cy, angle, length)
                        print(cx, cy)
                        exit()
                    if regression[img_idx, i, :].min() < 0:
                        print(regression[img_idx, i, :])
                        print(cx, cy, angle, length)
                        print(cx, cy)
                        exit()
        return regression

    def get_objectness_mask(self, gt: list[list[Line]]) -> torch.Tensor:
        objectness = torch.zeros((len(gt), self.n_columns, self.anchors_per_column, 1))

        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = gt[i][j].cx * self.n_columns // self.image_size[1]
                if x == self.n_columns:
                    x = self.n_columns - 1
                objectness[i, x, :, 0] = 1
        return objectness

    def get_lines_grouped_by_column(self, gt_image: list[Line]):
        out = [[] for _ in range(self.n_columns)]
        for i in gt_image:
            x = i.cx * self.n_columns // self.image_size[0]
            if x == self.n_columns:
                x = self.n_columns - 1
            out[x].append(i)
        return out