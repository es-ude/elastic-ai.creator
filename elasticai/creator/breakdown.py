"""Implementation of the breakdown. Used to represent multiple layers with
a fixed width.
"""
from typing import Union

import torch.nn
from torch.nn.utils import parametrize

from elasticai.creator.blocks import Conv2d_block
from elasticai.creator.layers import ChannelShuffle
from elasticai.creator.masks import fixedOffsetMask4D


def generate_conv2d_sequence_with_width(
    in_channels: int,
    out_channels: int,
    channel_width: int,
    kernel_size,
    weight_quantization,
    activation,
):
    """
    This will generate a sequential model composed of multiple layers, each a
    weight with channel width length.
    After the first block the channels are shuffled so the information of each
    output channel is composed from one in each input channel.
    eg: 256x256 1x1 convolution with channel width of 8. Each channel of the
     first layer will join information of 8 points  making 256/8 = 32 parts per out channel.
    So 32 groups and 256(out) = 256*32 output channels in the first layer. the
     second layer will again join 8  of those 32 resulting in 256*4
    groups and out channels. The last to keep the width will have a smaller
     number of groups e.g 256*4/8 = 128 groups. This has the effect of making very small kernels,
    being able to be represented by a much smaller amount of LUTs than a 256:1
     mapping.

    Args:
        in_channels:
        out_channels:
        channel_width:
        kernel_size:
        weight_quantization:
        activation:

    Returns:

    """
    layers: list[Union[Conv2d_block, ChannelShuffle]] = []
    if in_channels < channel_width:
        raise ValueError(
            "Channel width cannot be bigger than the number of input channels"
        )
    next_in_channels = in_channels
    next_groups = (in_channels) // channel_width
    next_out_channels = out_channels * next_groups

    while next_out_channels >= out_channels:
        layers.append(
            Conv2d_block(
                in_channels=next_in_channels,
                out_channels=next_out_channels,
                kernel_size=kernel_size,
                activation=activation,
                conv_quantizer=weight_quantization,
                groups=next_groups,
            )
        )
        if len(layers) == 1:
            layers.append(ChannelShuffle(groups=next_groups))
        if next_out_channels == out_channels:
            break
        next_in_channels = next_out_channels
        next_groups = (
            next_groups // channel_width
            if len(layers) > 2
            else next_out_channels // channel_width
        )
        next_out_channels = max(next_groups, out_channels)

    return torch.nn.Sequential(*layers)


class BreakdownConv2dBlock(torch.nn.Module):
    """
    Implementation of the kernel breakdown where each each channel will only
     select one row of the original nxn offset by
    a module of its own index.
    Args:
     conv_quantizer: The quantizer of the first QConv1d
     activation: an instance of the activation  after the first batch norm,
     pointwise_quantizer: the quantizer of the second Qconv1d
     pointwise_activation: an instance of the activation after the second
                           batch norm,
     pointwise_channel_width: the channel width of the pointwise convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
        activation,
        pointwise_activation,
        conv_quantizer=None,
        pointwise_quantizer=None,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):

        if isinstance(kernel_size, tuple):
            breakdown_multiplier = kernel_size[0] * groups
        else:
            breakdown_multiplier = kernel_size * groups
        super().__init__()
        self.Conv2d = Conv2d_block(
            in_channels=in_channels,
            out_channels=out_channels * breakdown_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            conv_quantizer=conv_quantizer,
        )
        if conv_quantizer is not None:
            parametrize.register_parametrization(
                self.depthwiseConv2d, "weight", conv_quantizer
            )
        mask = fixedOffsetMask4D(
            out_channels=out_channels * breakdown_multiplier,
            in_channels=in_channels,
            kernel_size=kernel_size,
            offset_axis=2,
            axis_width=1,
            groups=groups,
        )
        parametrize.register_parametrization(self.Conv2d.conv2d, "weight", mask)
        self.shuffle = ChannelShuffle(groups=groups)
        self.pointwiseConv2d = Conv2d_block(
            in_channels=out_channels * breakdown_multiplier,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            groups=out_channels,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            activation=pointwise_activation,
            conv_quantizer=pointwise_quantizer,
        )

    @property
    def codomain(self):
        if self.pointwise_activation is not None and hasattr(
            self.pointwise_activation, "codomain"
        ):
            return self.pointwise_activation.codomain
        return None

    def forward(self, input):
        x = self.Conv2d(input)
        x = self.shuffle(x)
        x = self.pointwiseConv2d(x)
        return x


class depthwisePointwiseBreakdownConv2dBlock(torch.nn.Module):
    """
    Sequence depthwise QConv2d - batchNorm -activation, pointwise_breakdown
    uses default batchNorm parameters. Most parameters affect the depthwise Qconv2d,
    Args:
     conv_quantizer: The weightr quantizer of the first Conv2d can be none
     activation: an instance of the activation  after the first batch norm,
     pointwise_quantizer: the quantizer of the pointwise Conv2d can be none
     pointwise_activation: an instance of the activation after each pointwise batch norm,
     pointwise_channel_width: the channel width of the pointwise convolution
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        activation,
        pointwise_activation,
        pointwise_channel_width,
        conv_quantizer=None,
        pointwise_quantizer=None,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.depthwiseConv2d = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
        )
        if conv_quantizer is not None:
            parametrize.register_parametrization(
                self.depthwiseConv2d, "weight", conv_quantizer
            )
        self.batchnorm = torch.nn.BatchNorm2d(in_channels)
        self.activation = activation

        self.pointwiseConv2d = generate_conv2d_sequence_with_width(
            in_channels,
            out_channels,
            pointwise_channel_width,
            1,
            pointwise_quantizer,
            pointwise_activation,
        )

    @property
    def codomain(self):
        if self.pointwise_activation is not None and hasattr(
            self.pointwise_activation, "codomain"
        ):
            return self.pointwise_activation.codomain
        return None

    def forward(self, input):
        x = self.depthwiseConv2d(input)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pointwiseConv2d(x)
        return x
