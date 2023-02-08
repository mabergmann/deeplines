import numpy as np
from torch.utils.data import Dataset
import random

from ..line import Line


class RandomLines(Dataset):
    def __init__(self, image_size: tuple[int, int], min_lines: int, max_lines: int):
        self.image_size = image_size
        self.min_lines = min_lines
        self.max_lines = max_lines

    def __getitem__(self, idx):
        img = np.zeros((self.image_size[0], self.image_size[1], 3))

        n_lines = random.randint(self.min_lines, self.max_lines)

        lines = []

        for i in range(n_lines):
            cx = random.randint(0, self.image_size[1])
            cy = random.randint(0, self.image_size[0])
            angle = random.random() * 2 * np.pi
            a_line = Line(cx=cx, cy=cy, angle=angle)
            lines.append(a_line)

        return img, lines
