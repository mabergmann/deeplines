import pytest
import torch

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


def test_one_line_correct(loss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    pred = torch.zeros((1, 9, 1))
    pred[0, 1, 0] = 1
    result = loss(pred, gt)
    assert result == 0


def test_one_line_incorrect(loss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    pred = torch.zeros((1, 9, 1))
    result = loss(pred, gt)
    assert result == pytest.approx(1 / 9)
