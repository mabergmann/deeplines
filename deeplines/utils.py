import math

import cv2
import numpy as np

from .line import Line


def draw_line(image: np.array, line: Line, color: tuple[int, int, int], thickness: int, confidence=False):
    p0 = (
        int(line.p0()[0]),
        int(line.p0()[1])
    )
    p1 = (
        int(line.p1()[0]),
        int(line.p1()[1])
    )
    image = cv2.line(image, p0, p1, color, thickness)
    if confidence:
        image = cv2.putText(image, f"{line.confidence}", p0, cv2.FONT_HERSHEY_SIMPLEX, 1, color)

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
        for n, c in enumerate(img_output):
            for line in c:
                objectness = float(line[0])
                if objectness >= threshold:
                    cx = float(line[1] * image_width)
                    cy = float(line[2] * image_height)
                    angle = float(line[3] * np.pi)
                    length = float(line[4] * image_width)

                    image_lines.append(Line(
                        cx=cx,
                        cy=cy,
                        length=length,
                        angle=angle,
                        confidence=objectness
                    ))

        batch_lines.append(image_lines)

    return batch_lines


def nms(lines, nms_threshold=25):
    batch_filtered_lines = []
    for image_lines in lines:
        filtered_image_lines = []
        image_lines.sort(key=lambda x: x.confidence, reverse=True)
        added_mask = np.zeros((len(image_lines)))
        distances = get_distance_between_lines(image_lines, image_lines)
        for n, line in enumerate(image_lines):
            if n == 0:  # No line added yet
                min_distance = float("inf")
            else:
                min_distance = min(distances[n, added_mask == 1])
            if min_distance >= nms_threshold:
                filtered_image_lines.append(line)
                added_mask[n] = 1
        batch_filtered_lines.append(filtered_image_lines)
    return batch_filtered_lines


def get_classifications_from_output(output):
    batch_classifications = output[:, :, 0].cpu().numpy()
    return batch_classifications


def draw_result(batch_images, batch_lines, batch_classifications):
    output = []
    for img, lines, classifications in zip(batch_images, batch_lines, batch_classifications):
        img_np = img.cpu().numpy().transpose((1, 2, 0)).copy()

        # Draw lines
        for conf in lines:
            img_np = draw_line(img_np, conf, (0, 0, 255), 2, confidence=False)

        img_np = cv2.resize(img_np, None, fx=5, fy=5)

        # # Draw classifications
        # for n in range(len(classifications)):
        #     x = int((n + 1) * img_np.shape[1] / len(classifications))
        #     h = img_np.shape[0]
        #     img_np = cv2.line(img_np, (x, 0), (x, h), (0, 0, 255), 2)
        # for n, conf in enumerate(classifications):
        #     conf = f"{float(conf):.2f}"
        #     cx = img_np.shape[1] * n / len(classifications)
        #     img_np = cv2.putText(img_np, conf, (int(cx), 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        output.append(img_np)

    return output
