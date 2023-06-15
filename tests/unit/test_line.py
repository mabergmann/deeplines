import math
import numpy as np
import pytest

from deeplines.line import Line


@pytest.fixture
def line():
    return Line(cx=100, cy=100, angle=0, length=50)


@pytest.fixture
def line2():
    return Line(cx=500, cy=590, angle=np.pi/2, length=200)


def test_creation(line):
    assert isinstance(line, Line)


def test_left(line):
    assert line.left() == 75


def test_right(line):
    assert line.right() == 125


def test_top(line2):
    assert line2.top() == 490


def test_bottom(line2):
    assert line2.bottom() == 690


def test_p0(line):
    expected = (
        line.cx - line.length * math.cos(line.angle) / 2,
        line.cy - line.length * math.sin(line.angle) / 2
    )
    calculated = line.p0()

    assert calculated[0] == expected[0]
    assert calculated[1] == expected[1]
