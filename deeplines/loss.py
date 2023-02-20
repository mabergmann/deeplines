import torch
from torch import nn


class DeepLineLoss(nn.Module):
    def __init__(self, image_size, n_columns):
        super().__init__()
        self.image_size = image_size
        self.n_columns = n_columns
        self.mse = torch.nn.MSELoss(reduction='mean')

    def forward(self, pred, gt):
        objectness = self.get_objectness_from_gt(gt)
        objectness = objectness.to(pred.device)
        return self.mse(pred, objectness)

    def get_objectness_from_gt(self, gt):

        objectness = torch.zeros((len(gt), self.n_columns, 1))

        for i in range(len(gt)):
            for j in range(len(gt[i])):
                x = gt[i][j].cx * self.n_columns // self.image_size[1]
                if x == self.n_columns:
                    x = self.n_columns - 1
                objectness[i, x, 0] = 1
        return objectness
