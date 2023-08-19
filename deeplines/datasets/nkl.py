import cv2
import math

import numpy as np
import pathlib as pl
import torch
from torch.utils.data import Dataset

from ..line import Line
from ..utils import draw_line


class NKL(Dataset):
    def __init__(self, image_size: tuple[int, int], split_file: str, images_folder: str):
        self.image_size = image_size
        with open(split_file) as f:
            self.images_filenames = f.readlines()
        self.dataset_folder = pl.Path(images_folder)

    def __len__(self) -> int:
        return len(self.images_filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, list[Line]]:
        filename = str(self.dataset_folder / self.images_filenames[idx].strip())
        ann_filename = filename.replace(".jpg", ".txt")
        img = cv2.imread(filename)
        original_shape = img.shape

        fx = self.image_size[1] / original_shape[1]
        fy = self.image_size[0] / original_shape[0]

        img = cv2.resize(img, self.image_size)

        with open(ann_filename) as f:
            ann = f.readline()

        ann = ann.split()
        n_lines = int(ann[0])

        lines = []

        for i in range(n_lines):
            p0 = (int(ann[i * 4 + 1]) * fx, int(ann[i * 4 + 2]) * fy)
            p1 = (int(ann[i * 4 + 3]) * fx, int(ann[i * 4 + 4]) * fy)
            cx = p1[0] + ((p0[0] - p1[0]) / 2)
            cy = p1[1] + ((p0[1] - p1[1]) / 2)

            v = (p0[0] - p1[0], p0[1] - p1[1]) if p0[1] > p1[1] else (p1[0] - p0[0], p1[1] - p0[1])
            length = np.linalg.norm(v)
            v = v / length

            angle = math.acos(v[0])
            a_line = Line(cx=cx, cy=cy, angle=angle, length=length)
            lines.append(a_line)

            # img = draw_line(img, a_line, (255, 255, 255), 3)

        img_th = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()

        return img_th, lines

    def collate_fn(self, batch: list[tuple[torch.Tensor, list[Line]]]) -> tuple[torch.Tensor, list[list[Line]]]:
        images = []
        lines = []

        for img, line in batch:
            images.append(img.unsqueeze(0))
            lines.append(line)

        images = torch.cat(images, dim=0)

        return images, lines
