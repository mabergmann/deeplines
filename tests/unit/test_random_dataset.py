import pytest
from torch.utils.data import Dataset

from deeplines.datasets.randomlines import RandomLines
from deeplines.line import Line


@pytest.fixture
def dataset():
    return RandomLines(image_size=(800, 800), min_lines=1, max_lines=5)


def test_dataloader_is_a_dataset(dataset):
    assert isinstance(dataset, Dataset)


def test_image_size(dataset):
    img, gt = dataset[0]
    assert img.shape == (800, 800, 3)


def test_correct_number_of_lines(dataset):
    img, gt = dataset[0]
    assert 1 <= len(gt) <= 5


def test_gt_is_list_of_lines(dataset):
    img, gt = dataset[0]
    for line in gt:
        assert isinstance(line, Line)
