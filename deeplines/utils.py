import math

import cv2
import numpy as np

from .line import Line


def draw_line(image: np.array, line: Line, color: tuple[int, int, int], thickness: int):
    p0 = (
            int(line.cx - line.length * math.cos(line.angle)),
            int(line.cy - line.length * math.sin(line.angle))
    )
    p1 = (
            int(line.cx + line.length * math.cos(line.angle)),
            int(line.cy + line.length * math.sin(line.angle))
    )
    image = cv2.line(image, p0, p1, color, thickness)

    return image
