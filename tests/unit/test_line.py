import pytest

from deeplines.line import Line


@pytest.fixture
def line():
    return Line(cx=100, cy=100, angle=0)


def test_creation(line):
    assert isinstance(line, Line)
