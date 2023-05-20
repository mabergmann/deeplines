import math
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ..line import Line
from ..utils import draw_line


class RandomLines(Dataset):
    def __init__(self, image_size: tuple[int, int], min_lines: int, max_lines: int):
        self.image_size = image_size
        self.min_lines = min_lines
        self.max_lines = max_lines

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        img = np.zeros((self.image_size[0], self.image_size[1], 3))

        n_lines = random.randint(self.min_lines, self.max_lines)

        lines = []

        for i in range(n_lines):
            cx = random.randint(0, self.image_size[1])
            cy = random.randint(0, self.image_size[0])
            angle = random.random() * np.pi
            max_length = self.maximum_length(cx, cy, angle)
            length = random.random() * max_length
            a_line = Line(cx=cx, cy=cy, angle=angle, length=length)
            lines.append(a_line)

            # try:
            #     assert a_line.left() >= 0
            #     assert a_line.top() >= 0
            #     assert a_line.right() <= self.image_size[0]
            #     assert a_line.bottom() <= self.image_size[1]
            # except:
            #     print(cx, cy, angle, length, max_length)
            #     exit()

            img = draw_line(img, a_line, (255, 255, 255), 3)

        img_th = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        return img_th, lines

    def maximum_length(self, cx, cy, angle):
        w, h = self.image_size

        # Calculate line length in both x and y directions
        x_length = abs(min(cx, w - cx) / math.cos(angle)) if math.cos(angle) != 0 else float("inf")
        y_length = abs(min(cy, h - cy) / math.sin(angle)) if math.sin(angle) != 0 else float("inf")

        # Return the minimum of the two lengths
        return min(x_length, y_length)

    def collate_fn(self, batch):
        images = []
        lines = []

        for img, line in batch:
            images.append(img.unsqueeze(0))
            lines.append(line)

        images = torch.cat(images, dim=0)

        return images, lines
