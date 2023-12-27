import torch
from torch import nn
from torchvision import models

from .pit import pit_b, pit_ti


class DeepLines(nn.Module):
    def __init__(self, n_columns: int, anchors_per_column: int, backbone_name: str):
        super().__init__()

        # init a pretrained resnet
        self.backbone_name = backbone_name
        print(backbone_name)
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights='DEFAULT')
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
        elif backbone_name == 'vgg16':
            backbone = models.vgg16(weights='DEFAULT')
            num_filters = 25088
            layers = list(backbone.children())[:-1]
            self.feature_extractor = nn.Sequential(*layers)
        elif backbone_name == 'pit_b':
            backbone = pit_b(pretrained=True)
            backbone.head = nn.Identity()
            num_filters = backbone.heads[-1] * backbone.base_dims[-1]
            self.feature_extractor = backbone
        elif backbone_name == 'pit_ti':
            backbone = pit_ti(pretrained=False)
            backbone.head = nn.Identity()
            num_filters = backbone.heads[-1] * backbone.base_dims[-1]
            self.feature_extractor = backbone
        else:
            raise NotImplementedError(f'Backbone {backbone_name} not implemented')

        self.classifier = nn.Sequential(
            nn.Linear(num_filters, n_columns * anchors_per_column * 5),
        )

        self.n_columns = n_columns
        self.anchors_per_column = anchors_per_column

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = x.reshape((x.shape[0], self.n_columns, self.anchors_per_column, -1))

        x[:, :, :, 0] = torch.sigmoid(x[:, :, :, 0])

        return x
