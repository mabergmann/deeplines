import numpy as np
import pytest
import torch

from deeplines.line import Line
from deeplines.loss import DeepLineLoss


@pytest.fixture
def loss():
    return DeepLineLoss(image_size=(800, 800), n_columns=9, anchors_per_column=5)


def test_get_objectness(loss: DeepLineLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]

    distances = np.zeros((1, 1, 45))
    distances[...] = 100
    distances[0, 0, 5] = 15

    objectness = loss.get_objectness_from_gt(gt, distances)
    for i in range(9):
        for j in range(5):
            if i == 1 and j == 0:
                assert objectness[0, i, j] == loss.distance_to_confidence(15)
            else:
                assert objectness[0, i, j] == 0


def test_get_objectness_multiple_lines(loss: DeepLineLoss):
    gt = [
        [
            Line(cx=100, cy=100, angle=0, length=50),
            Line(cx=200, cy=100, angle=30, length=42),
            Line(cx=100, cy=200, angle=60, length=28),
            Line(cx=200, cy=200, angle=90, length=200),
            Line(cx=123, cy=321, angle=42, length=13)
        ]
    ]

    distances = np.zeros((1, 5, 45))
    distances[...] = 100
    distances[0, 0, 5] = 15

    objectness = loss.get_objectness_from_gt(gt, distances)
    for i in range(9):
        for j in range(5):
            if i == 1 and j == 0:
                assert objectness[0, i, j] == loss.distance_to_confidence(15)
            else:
                assert objectness[0, i, j] == 0


def test_distance_to_confidence(loss: DeepLineLoss):
    assert loss.distance_to_confidence(10) == 0.5
    assert loss.distance_to_confidence(0) == 1
    assert loss.distance_to_confidence(float("inf")) == 0


def test_get_regression(loss: DeepLineLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    best_match_batch = [[gt[0][0]] * 45]
    regression = loss.get_regression_from_best_match(gt, best_match_batch)
    print(regression.shape)
    assert pytest.approx(regression[0, 1, 0, 0], 1e-4) == 100/800
    assert pytest.approx(regression[0, 1, 0, 1], 1e-4) == 100/800
    assert pytest.approx(regression[0, 1, 0, 2], 1e-4) == 0
    assert pytest.approx(regression[0, 1, 0, 3], 1e-4) == 50/800


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
    pred = torch.zeros((1, 9, 5, 5))
    pred[0, 1, :, 0] = 1
    pred[0, 1, :, 1] = 100/800
    pred[0, 1, :, 2] = 100/800
    pred[0, 1, :, 3] = 0
    pred[0, 1, :, 4] = 50/800
    result = loss(pred, gt)
    assert result["objectness"] == 0
