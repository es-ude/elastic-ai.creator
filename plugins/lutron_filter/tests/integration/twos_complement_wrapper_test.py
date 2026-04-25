import pytest
import torch
from elasticai.creator_plugins.lutron_filter.precompute.int_encoding_lutron_convolution import (
    LutronBitsToTwoComplementsWrapper,
)
from elasticai.creator_plugins.lutron_filter.precompute.truth_table_generation import (
    generate_input_tensor_1d as generate_input_tensor,
)
from torch.nn import Identity


class Wrapped(Identity):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] | int,
        groups: int = 1,
        stride: tuple[int, ...] | int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride,)


def test_zero_and_minus_one_for_single_bit():
    wrapped = Wrapped(1, 1, 1)
    wrapper = LutronBitsToTwoComplementsWrapper(num_bits=1, wrapped=wrapped)
    result = wrapper(torch.tensor([[-1, 1]]))
    assert result.tolist() == [[0, -1]]


def test_two_bit_values():
    wrapped = Wrapped(1, 1, 1)
    wrapper = LutronBitsToTwoComplementsWrapper(num_bits=2, wrapped=wrapped)
    result = wrapper(torch.tensor([[1, 1, -1, -1], [-1, 1, -1, 1]]))
    assert result.tolist() == [[-2, -1, 0, 1]]


def test_three_bit_values():
    wrapped = Wrapped(1, 1, 1)
    wrapper = LutronBitsToTwoComplementsWrapper(num_bits=3, wrapped=wrapped)
    result = wrapper(
        torch.tensor(
            [
                [1, 1, 1, 1, -1, -1, -1, -1],
                [-1, -1, 1, 1, -1, -1, 1, 1],
                [-1, 1, -1, 1, -1, 1, -1, 1],
            ]
        )
    )
    assert result.tolist() == [[-4, -3, -2, -1, 0, 1, 2, 3]]


@pytest.mark.parametrize(
    "in_channels, kernel_size, bits, expected",
    [
        (1, 1, 2, {(1, -1): (-2,), (1, 1): (-1,), (-1, -1): (0,), (-1, 1): (1,)}),
        (
            2,
            1,
            1,
            {(1, 1): (-1, -1), (-1, -1): (0, 0)},
        ),
        (
            1,
            2,
            1,
            {(1, 1): (-1, -1), (-1, -1): (0, 0), (-1, 1): (0, -1), (1, -1): (-1, 0)},
        ),
        (
            2,
            1,
            2,
            {
                (1, 1, 1, 1): (-1, -1),
                (-1, -1, -1, -1): (0, 0),
                (1, -1, 1, -1): (-2, -2),
                (-1, 1, -1, 1): (1, 1),
            },
        ),
    ],
)
def test_apply_to_lutron_conv(in_channels, kernel_size, bits, expected):
    wrapped = Wrapped(
        in_channels,
        1,
        kernel_size,
    )
    wrapper = LutronBitsToTwoComplementsWrapper(num_bits=bits, wrapped=wrapped)
    inputs = generate_input_tensor(
        wrapper.in_channels, kernel_size=wrapper.kernel_size, groups=wrapper.groups
    )
    outputs = wrapper(inputs)
    inputs = inputs.view(-1, in_channels * bits * kernel_size)
    outputs = outputs.view(-1, in_channels * kernel_size)
    inputs = inputs.tolist()
    outputs = outputs.tolist()
    io_pairs = dict((tuple(a), tuple(c)) for a, c in zip(inputs, outputs))
    assert io_pairs == expected
