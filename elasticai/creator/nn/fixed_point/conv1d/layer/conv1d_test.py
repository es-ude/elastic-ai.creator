from typing import cast

import pytest
import torch

from elasticai.creator.test_utils.temporary_file_structure import (
    get_savable_file_structure,
)

from .conv1d import Conv1d


@pytest.fixture
def conv1d() -> Conv1d:
    return Conv1d(
        total_bits=16,
        frac_bits=8,
        in_channels=3,
        out_channels=4,
        kernel_size=2,
        bias=False,
        signal_length=5,
    )


def conv_with_one_weights(signal_length: int) -> Conv1d:
    conv = Conv1d(
        total_bits=16,
        frac_bits=8,
        in_channels=1,
        out_channels=1,
        bias=False,
        signal_length=signal_length,
        kernel_size=2,
    )
    conv.weight.data = torch.ones_like(conv.weight)
    return conv


@pytest.fixture
def conv_with_limited_precision() -> Conv1d:
    return Conv1d(
        total_bits=4,
        frac_bits=1,
        kernel_size=2,
        signal_length=2,
        bias=False,
        in_channels=1,
        out_channels=1,
    )


@pytest.fixture
def batched_input() -> torch.Tensor:
    return torch.rand(3, 15)


def test_that_output_is_flat_for_unbatched_input(conv1d: Conv1d) -> None:
    input_data = torch.rand(15)
    prediction = conv1d(input_data)
    tensor_rank = prediction.dim()
    flat_tensor_rank = 1
    assert tensor_rank == flat_tensor_rank


def test_that_batch_dimension_is_kept(
    conv1d: Conv1d, batched_input: torch.Tensor
) -> None:
    prediction = conv1d(batched_input)
    batch_dimension = prediction.shape[0]
    expected_batch_dimension = batched_input.shape[0]
    assert batch_dimension == expected_batch_dimension


def test_that_batched_output_is_flat(
    conv1d: Conv1d, batched_input: torch.Tensor
) -> None:
    prediction = conv1d(batched_input)
    tensor_rank = prediction.dim()
    expected_rank = 2
    assert expected_rank == tensor_rank


@pytest.mark.parametrize(
    "signal_length, inputs, expected",
    [
        (2, [1.0, 1.0], [2.0]),
        (3, [1.0, 2.0, 4.0], [3.0, 6.0]),
    ],
)
def test_ones_weight_kernel_convolution(
    signal_length: int, inputs: list[float], expected: list[float]
) -> None:
    input_tensor = torch.tensor(inputs)
    prediction = conv_with_one_weights(signal_length)(input_tensor)
    assert expected == prediction.tolist()


def test_no_overflow_after_multiplication(conv_with_limited_precision: Conv1d) -> None:
    conv = conv_with_limited_precision
    conv.weight.data = 3.5 * torch.ones_like(conv.weight)
    prediction = conv(torch.tensor([3.5, -3]))
    expected = 1.5  # quantize(3.5 * 3.5 - 3 * 3.5) == quantize(1.75) == 1.5
    assert [expected] == prediction.tolist()


def test_no_underflow_after_multiplication(conv_with_limited_precision: Conv1d) -> None:
    conv = conv_with_limited_precision
    conv.weight.data = 0.5 * torch.ones_like(conv.weight)
    prediction = conv(torch.tensor([0.5, 0]))
    expected = 0  # quantize(0.5 * 0.5 + 0.5 * 0) == quantize(0.25) == 0
    assert [expected] == prediction.tolist()


@pytest.mark.parametrize(
    "inputs, expected",
    [([2.0, 2.0], [3.5]), ([-2.0, -3.0], [-4.0])],
)
def test_fxp_operations_additive_over_and_underflow(
    conv_with_limited_precision: Conv1d, inputs: list[float], expected: list[float]
) -> None:
    conv = conv_with_limited_precision
    conv.weight.data = torch.ones_like(conv.weight)
    input_tensor = torch.tensor(inputs)
    prediction = conv(input_tensor)
    assert expected == prediction.tolist()


def test_bias_addition() -> None:
    conv = Conv1d(
        total_bits=4,
        frac_bits=1,
        kernel_size=2,
        signal_length=2,
        bias=True,
        in_channels=1,
        out_channels=1,
    )
    conv.weight.data = torch.ones_like(conv.weight)
    conv.bias.data = 0.5 * torch.ones_like(conv.bias)  # type: ignore
    inputs = torch.tensor([0.0, 0.0])
    predictions = conv(inputs)
    assert predictions.tolist() == [0.5]


def test_conv1d_layer_creates_correct_design(conv1d: Conv1d) -> None:
    expected_conv1d_code = """-- Dummy File for testing implementation of conv1d Design
16
8
3
4
2
"""

    design = conv1d.create_design("conv1d")
    files = get_savable_file_structure(design)

    assert expected_conv1d_code == files["conv1d.vhd"]
