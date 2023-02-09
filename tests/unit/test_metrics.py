import pytest

from deeplines.metrics import MetricAccumulator
from deeplines.line import Line

@pytest.fixture
def metric_accumulator():
    return MetricAccumulator()


def test_is_instance(metric_accumulator):
    assert isinstance(metric_accumulator, MetricAccumulator)


def test_same_lines(metric_accumulator):
    detected_lines = [[Line(cx=10, cy=10, angle=0, length=10)]]
    gt_lines = [[Line(cx=10, cy=10, angle=0, length=10)]]

    metric_accumulator.update(detected_lines, gt_lines)

    assert metric_accumulator.get_recall() == 1
    assert metric_accumulator.get_precision() == 1
    assert metric_accumulator.get_f1() == 1
