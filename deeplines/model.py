import torch
from torch import nn
import torch.nn.functional as F


class DeepLines(nn.Module):
    def __init__(self, input_image_size, n_columns):
        super(DeepLines, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        new_size = (input_image_size[0]//16, input_image_size[1]//16)
        self.fc = nn.Linear(new_size[0] * new_size[1] * 512, n_columns)

        self.n_columns = n_columns

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc(x)
        x = x.reshape((x.shape[0], self.n_columns, -1))

        x = torch.sigmoid(x)

        return x
