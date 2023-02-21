import torch
from torch import nn


class DeepLineLoss(nn.Module):
    def __init__(self, image_size, n_columns):
        super().__init__()
        self.image_size = image_size
        self.n_columns = n_columns
        self.bce = torch.nn.BCELoss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, gt):
        objectness = self.get_objectness_from_gt(gt)
        objectness = objectness.to(pred.device)

        regression = self.get_regression_from_gt(gt)
        regression = regression.to(pred.device)

        loss_objectness = self.bce(pred[:, :, 0:1], objectness)
        loss_regression = self.mse(pred[:, :, 1:], regression)

        return (loss_objectness + loss_regression) / 2

    def get_objectness_from_gt(self, gt):

        objectness = torch.zeros((len(gt), self.n_columns, 1))

        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = gt[i][j].cx * self.n_columns // self.image_size[1]
                if x == self.n_columns:
                    x = self.n_columns - 1
                objectness[i, x, 0] = 1
        return objectness

    def get_regression_from_gt(self, gt):
        regression = torch.zeros((len(gt), self.n_columns, 4))

        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = gt[i][j].cx * self.n_columns // self.image_size[1]
                if x == self.n_columns:
                    x = self.n_columns - 1

                cx = x * self.image_size[0] / len(gt[i])
                cx += self.image_size[0] / (len(gt[i]) * 2)
                cy = self.image_size[0] / 2
                left = gt[i][j].left()
                right = gt[i][j].right()
                top = gt[i][j].top()
                bottom = gt[i][j].bottom()
                regression[i, j, 0] = (cx - left) / self.image_size[0]
                regression[i, j, 1] = (right - cx) / self.image_size[0]
                regression[i, j, 2] = (cy - top) / self.image_size[1]
                regression[i, j, 3] = (bottom - cy) / self.image_size[1]
        return regression

