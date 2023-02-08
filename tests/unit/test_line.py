import pytest

from deeplines.line import Line


@pytest.fixture
def line():
    return Line(cx=100, cy=100, angle=0, length=50)


def test_creation(line):
    assert isinstance(line, Line)
