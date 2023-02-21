import math

import cv2
import numpy as np

from .line import Line


def draw_line(image: np.array, line: Line, color: tuple[int, int, int], thickness: int, confidence=False):
    p0 = (
            int(line.cx - line.length * math.cos(line.angle)),
            int(line.cy - line.length * math.sin(line.angle))
    )
    p1 = (
            int(line.cx + line.length * math.cos(line.angle)),
            int(line.cy + line.length * math.sin(line.angle))
    )
    image = cv2.line(image, p0, p1, color, thickness)
    if confidence:
        image = cv2.putText(image, f"line.confidence", p0, cv2.FONT_HERSHEY_SIMPLEX, 1, color)

    return image


def euclidian_distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)


def euclidian_distance_lines(line1, line2):
    return euclidian_distance(line1.cx, line2.cx, line1.cy, line2.cy)


def get_distance_between_lines(lines1, lines2):
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
                )
            )
            distances[i, j] = distance
    return distances


def get_lines_from_output(output, image_width, image_height, threshold=.5):
    batch_lines = []
    for img_output in output:
        image_lines = []
        for n, l in enumerate(img_output):
            objectness = l[0]
            if objectness >= threshold:
                cx = n * image_width / len(img_output)
                cx += image_width / (len(img_output) * 2)  # add it to the center

                cy = image_height / 2

                left = cx - l[1] * image_width
                right = cx + l[2] * image_width
                top = cy - l[3] * image_height
                bottom = cy - l[4] * image_height

                cx = left + (right - left) / 2
                cy = top + (bottom - top) / 2

                length = math.sqrt((left - right) ** 2 + (top - bottom) ** 2)
                angle = math.acos((right-left)/length)

                image_lines.append(Line(
                    cx=cx,
                    cy=cy,
                    length=length,
                    angle=angle,
                    confidence=objectness
                ))
        batch_lines.append(image_lines)

    return batch_lines


def draw_result(batch_images, batch_lines):
    output = []
    for img, lines in zip(batch_images, batch_lines):
        img_np = img.cpu().numpy().transpose((1, 2, 0)).copy()
        for line in lines:
            img_np = draw_line(img_np, line, (0, 0, 255), 2, confidence=False)

        output.append(img_np)

    return output
