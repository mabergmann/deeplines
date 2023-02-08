import pytest

from deeplines.loss import DeepLineLoss
from deeplines.line import Line


@pytest.fixture
def loss():
    return DeepLineLoss(image_size=(800, 800), n_columns=9)


def test_get_objectness(loss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    objectness = loss.get_objectness_from_gt(gt)
    print(objectness)
    for i in range(9):
        if i == 1:
            assert objectness[0, i] == 1
        else:
            assert objectness[0, i] == 0
