from typing import Any

import pytest
import torch
from torch.nn import Conv1d as TorchConv1d

from elasticai.creator.base_modules.conv1d import Conv1d as CreatorConv1d
from elasticai.creator.base_modules.float_arithmetics import FloatArithmetics


def _set_fixed_params(conv: torch.nn.Conv1d) -> None:
    conv.weight.data = torch.ones_like(conv.weight.data)
    if conv.bias is not None:
        conv.bias.data = torch.ones_like(conv.bias.data)


def _torch_and_creator_conv1d_with_fixed_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
) -> tuple[TorchConv1d, CreatorConv1d]:
    conv_args: dict[str, Any] = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    torch_conv = TorchConv1d(**conv_args)
    creator_conv = CreatorConv1d(arithmetics=FloatArithmetics(), **conv_args)
    _set_fixed_params(torch_conv)
    _set_fixed_params(creator_conv)
    return torch_conv, creator_conv


def _ones_as_conv1d_input(
    batch_size: int, in_channels: int, input_length: int
) -> torch.Tensor:
    return (
        torch.ones(batch_size, in_channels, input_length)
        if batch_size > 0
        else torch.ones(in_channels, input_length)
    )


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,stride,padding,bias,input_length,batch_size",
    [
        (1, 3, 2, 1, 0, True, 10, 1),
        (1, 3, 2, 1, 0, True, 10, 0),
        (3, 1, 2, 1, 0, False, 10, 1),
        (3, 1, 3, 1, 0, True, 10, 4),
        (1, 3, 2, 1, 1, True, 10, 1),
    ],
)
def test_output_matches_expected(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    bias: bool,
    input_length: int,
    batch_size: int,
) -> None:
    torch_conv, creator_conv = _torch_and_creator_conv1d_with_fixed_params(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    inputs = _ones_as_conv1d_input(batch_size, in_channels, input_length)
    target_outputs = torch_conv(inputs)
    actual_outputs = creator_conv(inputs)
    assert (target_outputs == actual_outputs).all()
