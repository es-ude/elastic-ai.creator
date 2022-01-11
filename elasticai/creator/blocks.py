from torch import nn

from elasticai.creator.layers import QConv1d, QLinear

"""
Modules that work as a sequence of 3  or more layers. Useful for writing more compact models
"""


# when applying constraints to blocks loop with model.modules()
# finish with nn.Identity as an activation if not using Softmax
class QConv1d_block(nn.Module):
    """
    Sequence QConv1d - batchNorm - activation
    uses default batchNorm parameters. Most other parameters affect QConv1d
    @param conv_quantizer: The quantizer of the QConv1d
    @param activation: an instance of the activation
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        conv_quantizer,
        activation,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        constraints: list = None,
    ):
        super().__init__()

        self.conv1d = QConv1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_quantizer,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            constraints=constraints,
        )
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = activation

    @property
    def codomain(self):
        if self.activation is not None and hasattr(self.activation, "codomain"):
            return self.activation.codomain
        return None

    def forward(self, input):
        x = self.conv1d(input)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class QLinear_block(nn.Module):
    """
    Sequence QLinear - batchNorm - activation
    uses default batchNorm parameters. Most other parameters affect Qconv1d
    @param linear_quantizer: The quantizer of the QConv1d
    @param activation: an instance of the activation
    """

    def __init__(
        self,
        in_features,
        out_features,
        linear_quantizer,
        activation,
        bias=False,
        constraints: list = None,
    ):
        super().__init__()
        self.linear = QLinear(
            in_features,
            out_features,
            linear_quantizer,
            bias=bias,
            constraints=constraints,
        )
        self.batchnorm = nn.BatchNorm1d(out_features)
        self.activation = activation

    def forward(self, input):
        x = self.linear(input)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


class depthwiseQConv1d_block(nn.Module):
    """
    Sequence depthwise QConv1d - batchNorm -activation, 1x1 QConv1d - batchnorm - activation
    uses default batchNorm parameters. Most other parameters affect Qconv1d
    Args:
     conv_quantizer: The quantizer of the first QConv1d
     activation: an instance of the activation  after the first batch norm,
     pointwise_quantizer: the quantizer of the second Qconv1d
     pointwise_activation: an instance of the activation after the second batch norm,
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        conv_quantizer,
        pointwise_quantizer,
        activation,
        pointwise_activation,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        padding_mode="zeros",
        constraints: list = None,
        pointwise_constraints: list = None,
    ):
        super().__init__()

        self.depthwiseConv1d = QConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            quantizer=conv_quantizer,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding=padding,
            padding_mode=padding_mode,
            constraints=constraints,
        )
        self.batchnorm = nn.BatchNorm1d(in_channels)
        self.activation = activation

        self.pointwiseConv1d = QConv1d(
            in_channels,
            out_channels,
            1,
            pointwise_quantizer,
            stride=1,
            groups=1,
            bias=bias,
            constraints=pointwise_constraints,
        )
        self.pointwise_batchnorm = nn.BatchNorm1d(out_channels)
        self.pointwise_activation = pointwise_activation

    @property
    def codomain(self):
        if self.pointwise_activation is not None and hasattr(
            self.pointwise_activation, "codomain"
        ):
            return self.pointwise_activation.codomain
        return None

    def forward(self, input):
        x = self.depthwiseConv1d(input)
        x = self.batchnorm(x)
        x = self.activation(x)
        x = self.pointwiseConv1d(x)
        x = self.pointwise_batchnorm(x)
        x = self.pointwise_activation(x)
        return x
