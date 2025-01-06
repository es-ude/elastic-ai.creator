from collections import OrderedDict
from functools import partial

from torch.nn import Conv1d, Sequential

from elasticai.creator_plugins.lutron_filter.nn.lutron.binarize import Binarize
from elasticai.creator_plugins.lutron_filter.nn.lutron.conv_block import ConvBlock
from elasticai.creator_plugins.lutron_filter.torch.transformations import (
    split_convolutions,
)


def make_split_block(original: Conv1d, group_size):
    groups = original.in_channels // group_size
    in_channels = original.in_channels
    kernel_size = original.kernel_size
    out_channels = original.out_channels
    return Sequential(
        OrderedDict(
            first=ConvBlock(
                kernel_size=kernel_size,
                groups=groups,
                out_channels=groups * out_channels,
                stride=original.stride,
                in_channels=in_channels,
                activation=Binarize(),
            ),
            second=original,
        )
    )


def test_split_transform():
    m = Sequential(OrderedDict(conv0=ConvBlock(6, 6, 6, activation=Binarize())))
    gm = split_convolutions(m, partial(make_split_block, group_size=1))
    actual = gm.code
    expected = """\n\n\ndef forward(self, input):
    input_1 = input
    conv0_conv = self.conv0.conv_split(input_1);  input_1 = None
    conv0_mpool = self.conv0.mpool(conv0_conv);  conv0_conv = None
    conv0_bn = self.conv0.bn(conv0_mpool);  conv0_mpool = None
    conv0_activation = self.conv0.activation(conv0_bn);  conv0_bn = None
    return conv0_activation
    """
    assert expected == actual
