import numpy as np
import pytest
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

    l1 = Line(cx=10, cy=100, angle=0, length=100)
    l2 = Line(cx=100, cy=100, angle=0, length=100)

    assert utils.get_distance_between_lines([l1], [l2])[0, 0] == 90


def test_get_lines_from_output_1():
    output = torch.zeros((1, 9, 5))
    output[0, 4, 0] = 1

    lines = utils.get_lines_from_output(output, 224, 224)

    assert len(lines) == 1
    assert len(lines[0]) == 1
    assert lines[0][0].cx == 112


def test_get_lines_from_output_2():
    output = torch.zeros((1, 9, 5))
    output[0, 1, 0] = 1
    output[0, 1, 1] = 58.3333333/800
    output[0, 1, 2] = -8.3333333/800
    output[0, 1, 3] = 300/800
    output[0, 1, 4] = -300/800

    lines = utils.get_lines_from_output(output, 800, 800)

    assert len(lines) == 1
    assert len(lines[0]) == 1
    assert pytest.approx(lines[0][0].cx) == 100
    assert pytest.approx(lines[0][0].cy) == 100
    assert pytest.approx(lines[0][0].angle) == 0
    assert pytest.approx(lines[0][0].length) == 50
