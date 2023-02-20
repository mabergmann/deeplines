import torch
from torch import nn
from torchvision import models


class DeepLines(nn.Module):
    def __init__(self, n_columns):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Linear(num_filters, n_columns)
        self.n_columns = n_columns

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = x.reshape((x.shape[0], self.n_columns, -1))

        x = torch.sigmoid(x)

        return x
