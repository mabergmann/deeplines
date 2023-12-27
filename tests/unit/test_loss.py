import numpy as np
import pytest
import torch

from deeplines.line import Line
from deeplines.losses.hausdorff_loss import HaussdorffLoss


@pytest.fixture
def loss():
    return HaussdorffLoss(image_size=(800, 800), n_columns=9, anchors_per_column=5)


def test_get_objectness(loss: HaussdorffLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]

    distances = np.zeros((1, 45))
    distances[...] = 100
    distances[0, 5] = 15

    objectness = loss.get_objectness_from_gt(gt, distances)
    for i in range(9):
        for j in range(5):
            if i == 1 and j == 0:
                assert objectness[0, i, j] == loss.distance_to_confidence(15)
            else:
                assert objectness[0, i, j] == 0


def test_get_objectness_multiple_lines(loss: HaussdorffLoss):
    gt = [
        [
            Line(cx=100, cy=100, angle=0, length=50),
            Line(cx=200, cy=100, angle=30, length=42),
            Line(cx=100, cy=200, angle=60, length=28),
            Line(cx=200, cy=200, angle=90, length=200),
            Line(cx=123, cy=321, angle=42, length=13)
        ]
    ]

    distances = np.zeros((1, 45))
    distances[...] = 100
    distances[0, 5] = 15

    objectness = loss.get_objectness_from_gt(gt, distances)
    for i in range(9):
        for j in range(5):
            if i == 1 and j == 0:
                assert objectness[0, i, j] == loss.distance_to_confidence(15)
            else:
                assert objectness[0, i, j] == 0


def test_distance_to_confidence(loss: HaussdorffLoss):
    assert loss.distance_to_confidence(10) == 0.5
    assert loss.distance_to_confidence(0) == 1
    assert loss.distance_to_confidence(float("inf")) == 0


def test_returns_dict(loss: HaussdorffLoss):
    gt = [[Line(cx=100, cy=100, angle=0, length=50)]]
    pred = torch.zeros((1, 9, 5, 5))
    pred[0, 1, :, 0] = 1
    pred[0, 1, :, 1] = 100/800
    pred[0, 1, :, 2] = 100/800
    pred[0, 1, :, 3] = 0
    pred[0, 1, :, 4] = 50/800
    result = loss(pred, gt)

    assert isinstance(result, dict)


def test_get_points_from_pred_shape(loss: HaussdorffLoss):
    pred = torch.zeros((1, 9, 5, 5))
    pred[0, 1, :, 0] = 1
    pred[0, 1, :, 1] = 100/800
    pred[0, 1, :, 2] = 100/800
    pred[0, 1, :, 3] = 0
    pred[0, 1, :, 4] = 50/800

    p0_x, p0_y, p1_x, p1_y = loss.get_points_from_pred(pred)
    assert p0_x.shape == (1, 9, 5)
    assert p0_y.shape == (1, 9, 5)
    assert p1_x.shape == (1, 9, 5)
    assert p1_y.shape == (1, 9, 5)


def test_euclidean_distance(loss: HaussdorffLoss):
    p0_x = torch.Tensor([[[30, 60, 90], [0, 0, 0]]])
    p0_y = torch.Tensor([[[40, 80, 120], [30, 15, 3]]])
    p1_x = torch.Tensor([[[0, 0, 0], [40, 20, 4]]])
    p1_y = torch.Tensor([[[0, 0, 0], [0, 0, 0]]])

    expected = torch.Tensor([[[50, 100, 150], [50, 25, 5]]])

    distance = loss.get_euclidean_distance(p0_x, p0_y, p1_x, p1_y)

    assert (expected == distance).all()
