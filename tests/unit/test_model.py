import pytest
import torch
from torch import nn

from deeplines.model import DeepLines


@pytest.fixture
def model():
    return DeepLines(9, 5, "resnet50")


def test_model_is_nn_module(model):
    assert isinstance(model, nn.Module)


# def test_expected_output_shape(model):
#     input_batch = torch.zeros((1, 3, 800, 800))
#     with torch.no_grad():
#         output = model(input_batch)
#     assert output.shape == torch.Size([1, 9, 5, 5])
