import random

import numpy as np
from torch.utils.data import Dataset


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
            lines.append(None)

        return img, lines
