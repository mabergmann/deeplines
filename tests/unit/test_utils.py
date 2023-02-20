import numpy as np
import torch

from deeplines import utils
from deeplines.line import Line


def test_draw_line():
    img = np.zeros((800, 800, 3))
    line = Line(cx=100, cy=100, angle=0, length=50)

    img = utils.draw_line(img, line, (255, 255, 255), 3)

    assert (img[100, 100] == [255, 255, 255]).all()
    assert (img[100, 51] == [255, 255, 255]).all()
    assert (img[100, 140] == [255, 255, 255]).all()

    assert (img[50, 100] == [0, 0, 0]).all()
    assert (img[100, 200] == [0, 0, 0]).all()


def test_line_distance():
    l1 = Line(cx=100, cy=100, angle=0, length=100)
    l2 = Line(cx=100, cy=100, angle=0, length=100)

    assert utils.get_distance_between_lines([l1], [l2])[0, 0] == 0


def test_get_lines_from_output():
    output = torch.zeros((1, 9, 1))
    output[0, 4, 0] = 1

    lines = utils.get_lines_from_output(output, 224, 224)

    assert len(lines) == 1
    assert len(lines[0]) == 1
    assert lines[0][0].cx == 112
