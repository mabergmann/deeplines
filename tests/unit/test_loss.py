import numpy as np
import pytest
import torch

from deeplines.line import Line
from deeplines.loss import DeepLineLoss


@pytest.fixture
def loss():
    return DeepLineLoss(image_size=(800, 800), n_columns=9)


def test_get_objectness(loss: DeepLineLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    objectness = loss.get_objectness_from_gt(gt)
    for i in range(9):
        if i == 1:
            assert objectness[0, i] == 1
        else:
            assert objectness[0, i] == 0


def test_get_regression(loss: DeepLineLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    best_match_batch = [[gt[0][0]] * 9]
    regression = loss.get_regression_from_best_match(gt, best_match_batch)
    assert pytest.approx(regression[0, 1, 0], 1e-4) == 58.3333/800
    assert pytest.approx(regression[0, 1, 1], 1e-4) == -8.3333/800
    assert pytest.approx(regression[0, 1, 2], 1e-4) == 300/800
    assert pytest.approx(regression[0, 1, 3], 1e-4) == -300/800


# def test_get_regression2(loss: DeepLineLoss):
#     gt = [[Line(cx=100, cy=100, angle=2*np.pi/3, length=50)]]
#     best_match_batch = [[gt[0][0]] * 9]
#     regression = loss.get_regression_from_best_match(gt, best_match_batch)
#     assert pytest.approx(regression[0, 1, 0], 1e-4) == 58.3333/800
#     assert pytest.approx(regression[0, 1, 1], 1e-4) == -8.3333/800
#     assert pytest.approx(regression[0, 1, 2], 1e-4) == 300/800
#     assert pytest.approx(regression[0, 1, 3], 1e-4) == -300/800


def test_one_line_correct(loss: DeepLineLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    pred = torch.zeros((1, 9, 5))
    pred[0, 1, 0] = 1
    pred[0, 1, 1] = 58.3333333/800
    pred[0, 1, 2] = -8.3333333/800
    pred[0, 1, 3] = 300/800
    pred[0, 1, 4] = -300/800
    result = loss(pred, gt)
    assert result["objectness"] == 0


def test_one_line_class_incorrect(loss: DeepLineLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    pred = torch.zeros((1, 9, 5))
    result = loss(pred, gt)
    assert result["objectness"] == pytest.approx(100 / 9)
