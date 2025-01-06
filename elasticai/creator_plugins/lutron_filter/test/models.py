from collections import OrderedDict
from functools import partial

import torch
from torch import nn as nn
from torch.nn import BatchNorm1d, Conv1d, Identity, MaxPool1d, Module

from elasticai.creator.ir.helpers import FilterParameters
from elasticai.creator_plugins.lutron_filter.torch.lutron_modules import LutronConv


class HumbleBinarization(Module):
    def forward(self, x):
        return torch.sign(x)


class SmoothSigmoid(Module):
    def __init__(self):
        super().__init__()
        self.temperature = 4.0

    def forward(self, x):
        return torch.sigmoid(x / self.temperature)


def make_dw_conv_block(first, second, kernel_size, in_channels, out_channels, stride=1):
    return torch.nn.Sequential(
        OrderedDict(
            depthwise=first(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                stride=stride,
            ),
            pointwise=second(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=out_channels,
            ),
        )
    )


class ConvBlock(Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        pooling: MaxPool1d | Identity = Identity(),
        groups: int = 1,
        bias: bool = True,
        activation: Module = Identity(),
    ):
        super().__init__()
        self.bn = BatchNorm1d(out_channels)
        self.conv = Conv1d(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            groups=groups,
            bias=bias,
        )
        self.mpool = pooling
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.mpool(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


def lutron_dw_conv_block(
    kernel_size,
    in_channels,
    out_channels,
    pooling_window: int | None = None,
    stride: int = 1,
    pooling_stride: int | None = None,
):
    first = partial(ConvBlock, activation=HumbleBinarization())
    if pooling_stride is None:
        pooling_stride = pooling_window
    if pooling_window is not None:
        second = partial(
            ConvBlock,
            activation=HumbleBinarization(),
            pooling=MaxPool1d(kernel_size=pooling_window, stride=pooling_stride),
        )
    else:
        second = partial(ConvBlock, activation=HumbleBinarization())
    return make_dw_conv_block(
        first,
        second,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
    )


class Model(Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.block0 = LutronConv(
            lutron_dw_conv_block(
                in_channels=1, out_channels=2, kernel_size=1, pooling_window=1
            ),
            filter_parameters=FilterParameters(
                in_channels=1, out_channels=2, kernel_size=1, stride=1, groups=1
            ),
        )
        self.block1 = LutronConv(
            lutron_dw_conv_block(in_channels=2, out_channels=1, kernel_size=1),
            filter_parameters=FilterParameters(
                in_channels=2, out_channels=1, kernel_size=1, stride=1, groups=1
            ),
        )
        self.num_blocks = num_blocks

    def forward(self, x):
        x = self.block0(x)
        if self.num_blocks > 1:
            x = self.block1(x)
        return x


class ModelWithBinaryQuantization(Module):
    def __init__(self, num_blocks: int):
        super().__init__()
        self.block0 = LutronConv(
            lutron_dw_conv_block(
                in_channels=1,
                out_channels=2,
                kernel_size=1,
                pooling_window=1,
            ),
            filter_parameters=FilterParameters(
                in_channels=1, out_channels=2, kernel_size=1, stride=1, groups=1
            ),
        )
        self.block1 = LutronConv(
            lutron_dw_conv_block(in_channels=2, out_channels=1, kernel_size=1),
            filter_parameters=FilterParameters(
                in_channels=2, out_channels=1, kernel_size=1, stride=1, groups=1
            ),
        )
        self.num_blocks = num_blocks

    def forward(self, x):
        x = self.block0(x)
        if self.num_blocks > 1:
            x = self.block1(x)
        return x


class MaxPoolModelForLutronBlocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpool = nn.MaxPool1d(kernel_size=2)
        self.quant = HumbleBinarization()
        self.conv = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1)
        self.bn = nn.BatchNorm1d(num_features=2)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.quant(x)
        x = self.mpool(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.quant(x)
        x = self.mpool(x)
        return x
