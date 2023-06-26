import torch
from torch import nn
from torchvision import models


class DeepLines(nn.Module):
    def __init__(self, n_columns: int, anchors_per_column: int, backbone_name: str):
        super().__init__()

        # init a pretrained resnet
        print(backbone_name)
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights='DEFAULT')
            num_filters = backbone.fc.in_features
            layers = list(backbone.children())[:-1]
        elif backbone_name == 'vgg16':
            backbone = models.vgg16(weights='DEFAULT')
            num_filters = 25088
            layers = list(backbone.children())[:-1]
        else:
            raise NotImplementedError(f'Backbone {backbone_name} not implemented')
        self.feature_extractor = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            # nn.Linear(num_filters, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, 4096),
            # nn.ReLU(inplace=True),
            # nn.Linear(4096, n_columns * 5),
            nn.Linear(num_filters, n_columns * anchors_per_column * 5),
        )

        # self.classifier = nn.Linear(num_filters, n_columns * 5)
        self.n_columns = n_columns
        self.anchors_per_column = anchors_per_column

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        x = x.reshape((x.shape[0], self.n_columns, self.anchors_per_column, -1))

        x[:, :, :, 0] = torch.sigmoid(x[:, :, :, 0])
        # x[:, :, :3] = torch.sigmoid(x[:, :, :3])
        # x = torch.sigmoid(x)

        return x
