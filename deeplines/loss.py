import numpy as np
import torch
from torch import nn

from . import utils


class DeepLineLoss(nn.Module):
    def __init__(self, image_size, n_columns):
        super().__init__()
        self.image_size = image_size
        self.n_columns = n_columns
        self.bce = torch.nn.BCELoss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, gt):

        lines_batch = utils.get_lines_from_output(pred, self.image_size[0], self.image_size[1], threshold=0)
        distances_batch = []
        best_match_batch = []

        for gt_image, lines in zip(gt, lines_batch):
            assert len(lines) == self.n_columns
            distances = utils.get_distance_between_lines(gt_image, lines)
            distances_batch.append(distances)

            best_match_image = []

            for n, i in enumerate(lines):
                best_match_idx = np.argmin(distances[:, n])
                best_match_image.append(gt_image[best_match_idx])
            best_match_batch.append(best_match_image)

        objectness = self.get_objectness_from_gt(gt)
        objectness = objectness.to(pred.device)

        regression = self.get_regression_from_best_match(lines_batch, best_match_batch)
        regression = regression.to(pred.device)

        loss_objectness = 0.25 * self.bce(pred[:, :, 0:1], objectness)
        loss_center = 0.25 * self.mse(objectness * pred[:, :, 1:3], objectness * regression[:, :, :2])
        loss_angle = 0.25 * self.mse(objectness * pred[:, :, 3:4], objectness * regression[:, :, 2:3])
        loss_length = 0.25 * self.mse(objectness * pred[:, :, 4:], objectness * regression[:, :, 3:])
        assert regression[:, :, 3:].min() >= 0
        assert regression[:, :, 3:].max() <= 1

        loss = {
            "center": loss_center,
            "angle": loss_angle,
            "length": loss_length,
            "objectness": loss_objectness
        }

        return loss

    def get_objectness_from_gt(self, gt):

        objectness = torch.zeros((len(gt), self.n_columns, 1))

        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = gt[i][j].cx * self.n_columns // self.image_size[1]
                if x == self.n_columns:
                    x = self.n_columns - 1
                objectness[i, x, 0] = 1
        return objectness

    def get_regression_from_best_match(self, lines_batch, best_match_batch):
        regression = torch.zeros((len(lines_batch), self.n_columns, 4))

        for img_idx in range(len(lines_batch)):
            for i in range(self.n_columns):
                cx = best_match_batch[img_idx][i].cx
                cy = best_match_batch[img_idx][i].cy
                angle = best_match_batch[img_idx][i].angle
                length = best_match_batch[img_idx][i].length

                regression[img_idx, i, 0] = cx / self.image_size[0]
                regression[img_idx, i, 1] = cy / self.image_size[1]
                regression[img_idx, i, 2] = angle / np.pi
                regression[img_idx, i, 3] = length / self.image_size[0]
                if regression[img_idx, i, :].max() > 1:
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

    # def get_regression_from_best_match(self, lines_batch, best_match_batch):
    #     regression = torch.zeros((len(lines_batch), self.n_columns, 4))

    #     for img_idx in range(len(lines_batch)):
    #         for i in range(self.n_columns):
    #             cx = i * self.image_size[0] / self.n_columns
    #             cx += self.image_size[0] / (self.n_columns * 2)  # add it to the center

    #             cy = self.image_size[1] / 2

    #             p0 = best_match_batch[img_idx][i].p0()
    #             p1 = best_match_batch[img_idx][i].p1()

    #             if p0[0] > p1[0]:
    #                 p0, p1 = p1, p0

    #             left = p0[0]
    #             right = p1[0]
    #             top = p0[1]
    #             bottom = p1[1]
    #             regression[img_idx, i, 0] = (cx - left) / self.image_size[0]
    #             regression[img_idx, i, 1] = (right - cx) / self.image_size[0]
    #             regression[img_idx, i, 2] = (cy - top) / self.image_size[1]
    #             regression[img_idx, i, 3] = (bottom - cy) / self.image_size[1]
    #             # if regression[img_idx, i, :].max() > 1:
    #             #     print(regression[img_idx, i, :])
    #             #     print(left, right, top, bottom)
    #             #     print(cx, cy)
    #             #     exit()
    #     return regression
