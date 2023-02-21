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


def euclidian_distance(line1, line2):
    return math.sqrt((line1.cx - line2.cx) ** 2 + (line1.cy - line2.cy) ** 2)


def get_distance_between_lines(lines1, lines2):
    distances = np.zeros((len(lines1), len(lines2)))
    for i, l1 in enumerate(lines1):
        for j, l2 in enumerate(lines2):
            distances[i, j] = euclidian_distance(l1, l2)
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
                image_lines.append(Line(
                    cx=cx,
                    cy=image_height/2,
                    length=10,
                    angle=0,
                    confidence=objectness
                ))
        batch_lines.append(image_lines)

    return batch_lines


def draw_result(batch_images, pred_batch):
    output = []
    for img, pred in zip(batch_images, pred_batch):
        img_np = img.cpu().numpy().transpose((1, 2, 0)).copy()
        img_np = cv2.resize(img_np, None, fx=5, fy=5)
        for n in range(len(pred)):
            x = int((n + 1) * img_np.shape[1] / len(pred))
            h = img_np.shape[0]
            img_np = cv2.line(img_np, (x, 0), (x, h), (0, 0, 255), 2)
        for n, line in enumerate(pred):
            conf = f"{float(line[0]):.2f}"
            cx = img_np.shape[1] * n / len(pred)
            img_np = cv2.putText(img_np, conf, (int(cx), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        output.append(img_np)

    return output
