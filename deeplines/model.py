import torch
from torch import nn
from torchvision import models


class DeepLines(nn.Module):
    def __init__(self, n_columns, backbone):
        super().__init__()

        # init a pretrained resnet
        if backbone == "resnet50":
            backbone = models.resnet50(weights="DEFAULT")
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
        elif backbone == "vgg16":
            backbone = models.vgg16(weights="DEFAULT")
            num_filters = 25088
            layers = list(backbone.children())[:-1]
        else:
            raise NotImplementedError(f"Backbone {backbone} not implemented")
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Linear(num_filters, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_columns),
        )
        self.n_columns = n_columns

    def forward(self, x):
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = x.reshape((x.shape[0], self.n_columns, -1))

        x = torch.sigmoid(x)

        return x
