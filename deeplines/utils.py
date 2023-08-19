import math

import cv2
import numpy as np
import torch

from .line import Line


def draw_line(image: np.array, line: Line, color: tuple[int, int, int], thickness: int, confidence: bool = False) -> np.array:
    p0 = (
        int(line.p0()[0]),
        int(line.p0()[1]),
    )
    p1 = (
        int(line.p1()[0]),
        int(line.p1()[1]),
    )
    image = cv2.line(image, p0, p1, color, thickness)
    if confidence:
        image = cv2.putText(image, f'{line.confidence}', p0, cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    return image


def euclidian_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def euclidian_distance_lines(line1: Line, line2: Line) -> float:
    return euclidian_distance(line1.cx, line2.cx, line1.cy, line2.cy)


def get_distance_between_lines(lines1: list[Line], lines2: list[Line]) -> np.array:
    distances = np.zeros((len(lines1), len(lines2)))
    for i, l1 in enumerate(lines1):
        for j, l2 in enumerate(lines2):
            l1p0 = l1.p0()
            l1p1 = l1.p1()
            l2p0 = l2.p0()
            l2p1 = l2.p1()
            distance = max(
                min(
                    euclidian_distance(l1p0[0], l1p0[1], l2p0[0], l2p0[1]),
                    euclidian_distance(l1p0[0], l1p0[1], l2p1[0], l2p1[1]),
                ),
                min(
                    euclidian_distance(l1p1[0], l1p1[1], l2p0[0], l2p0[1]),
                    euclidian_distance(l1p1[0], l1p1[1], l2p1[0], l2p1[1]),
                ),
                min(
                    euclidian_distance(l1p0[0], l1p0[1], l2p0[0], l2p0[1]),
                    euclidian_distance(l1p1[0], l1p1[1], l2p0[0], l2p0[1]),
                ),
                min(
                    euclidian_distance(l1p0[0], l1p0[1], l2p1[0], l2p1[1]),
                    euclidian_distance(l1p1[0], l1p1[1], l2p1[0], l2p1[1]),
                ),
            )
            distances[i, j] = distance
    return distances


def get_lines_from_output(output: torch.Tensor, image_width: int, image_height: int, threshold: float = .5) -> list[list[Line]]:
    batch_lines = []
    for img_output in output:
        image_lines = []
        for n, c in enumerate(img_output):
            for line in c:
                objectness = float(line[0])
                if objectness >= threshold:
                    cx = float(line[1] * image_width)
                    cy = float(line[2] * image_height)
                    angle = float(line[3] * np.pi)
                    length = float(line[4] * image_width)

                    image_lines.append(
                        Line(
                            cx=cx,
                            cy=cy,
                            length=length,
                            angle=angle,
                            confidence=objectness,
                        ),
                    )

        batch_lines.append(image_lines)

    return batch_lines


def nms(lines: list[list[Line]], nms_threshold: float = 25) -> list[list[Line]]:
    batch_filtered_lines = []
    for image_lines in lines:
        filtered_image_lines = []
        image_lines.sort(key=lambda x: x.confidence, reverse=True)
        added_mask = np.zeros(len(image_lines))
        distances = get_distance_between_lines(image_lines, image_lines)
        for n, line in enumerate(image_lines):
            if n == 0:  # No line added yet
                min_distance = float('inf')
            else:
                min_distance = min(distances[n, added_mask == 1])
            if min_distance >= nms_threshold:
                filtered_image_lines.append(line)
                added_mask[n] = 1
        batch_filtered_lines.append(filtered_image_lines)
    return batch_filtered_lines


def get_classifications_from_output(output: torch.Tensor) -> np.array:
    batch_classifications = output[:, :, 0].cpu().numpy()
    return batch_classifications


def out_to_images(batch_images: torch.Tensor) -> list[np.array]:
    output = []
    for img in batch_images:
        img_np = img.cpu().numpy().transpose((1, 2, 0)).copy()
        output.append(img_np)

    return output


def draw_result(batch_images: list[np.array], batch_lines: list[list[Line]], color: tuple[int, int, int]) -> list[np.array]:
    output = []
    for img_np, lines in zip(batch_images, batch_lines):
        # Draw lines
        for conf in lines:
            img_np = draw_line(img_np, conf, color, 2, confidence=False)

        output.append(img_np)

    return output
