from collections import OrderedDict

from torch.nn import BatchNorm1d, Conv1d, Identity, MaxPool1d, Module, Sequential


def make_dw_conv_block(first, second, kernel_size, in_channels, out_channels, stride=1):
    return Sequential(
        OrderedDict(
            depthwise=first(
                kernel_size=kernel_size,
                in_channels=in_channels,
                out_channels=in_channels,
                groups=in_channels,
                stride=stride,
            ),
            pointwise=second(
                kernel_size=1,
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
            padding=0,
        )
        self.mpool = pooling
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.mpool(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
