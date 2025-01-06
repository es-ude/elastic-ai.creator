from functools import partial

from torch.nn import MaxPool1d

from .binarize import Binarize
from .conv_block import (
    ConvBlock,
    make_dw_conv_block,
)


def depthwise_separable_block(
    kernel_size,
    in_channels,
    out_channels,
    pooling_window: int | None = None,
    stride: int = 1,
    pooling_stride: int | None = None,
):
    binarization = Binarize
    first = partial(ConvBlock, activation=binarization())
    if pooling_stride is None:
        pooling_stride = pooling_window
    if pooling_window is not None:
        second = partial(
            ConvBlock,
            activation=binarization(),
            pooling=MaxPool1d(kernel_size=pooling_window, stride=pooling_stride),
        )
    else:
        second = partial(ConvBlock, activation=binarization())
    return make_dw_conv_block(
        first,
        second,
        kernel_size=kernel_size,
        in_channels=in_channels,
        out_channels=out_channels,
        stride=stride,
    )
