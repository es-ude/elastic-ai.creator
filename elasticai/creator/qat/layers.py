import types
from abc import abstractmethod
from itertools import product
from typing import Any, Callable, Optional, Protocol

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Conv1d, Module, Parameter
from torch.nn.utils.parametrize import register_parametrization

from elasticai.creator.mlframework import Module, Tensor
from elasticai.creator.precomputation.input_domains import (
    create_codomain_for_1d_conv,
    create_codomain_for_depthwise_1d_conv,
)
from elasticai.creator.precomputation.precomputation import precomputable
from elasticai.creator.qat.functional import binarize as BinarizeFn
from elasticai.creator.tags_utils import TaggedModule

"""Implementation of quantizers and quantized variants of pytorch layers"""
Quantize = Module


class Binarize(torch.nn.Module):
    """
    Implementation of binarizer with possible bits [-1,1]"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return BinarizeFn.apply(x)

    @staticmethod
    def right_inverse(x: Tensor) -> Tensor:
        return x


class Ternarize(torch.nn.Module):
    """
    Implementation of ternarizer with possible bits [-1,0,1]
    Args:
     zero_window_width: all numbers x with |x| <= |zero_window_width| will be rounded to zero, *default* value is
     0.5 which corresponds to natural rounding
    """

    def __init__(
        self,
        zero_window_width=0.5,
        trainable=False,
    ):
        super().__init__()

        factor = torch.tensor(0.5 / zero_window_width)

        # naming only possible per axis and is only a prototype feature --> decline
        # requires_grad flag to false makes parameters non-trainable
        self.widening_factor = torch.nn.Parameter(factor, requires_grad=trainable)

    def forward(self, x):
        clipped = torch.clamp(self.widening_factor * x, min=-1, max=1)
        rounded = torch.round(clipped)
        with torch.no_grad():
            no_gradient = rounded - clipped
        return clipped + no_gradient

    @staticmethod
    def right_inverse(x: Tensor) -> Tensor:
        return x


class ResidualQuantization(torch.nn.Module):
    """
    Implementation of residual quantization similar to Rebnet: https://arxiv.org/pdf/1711.01243.pdf
    But instead of summing the bits, they are concatenated, multiplying the second dimension by n bits.
    Here the encoded form is used to facilitate precalculation.
    Args:
     num_bits: number of residual quantization levels or bits
    """

    def __init__(self, num_bits=2):
        super().__init__()
        self.num_bits = num_bits
        self.weight = torch.nn.Parameter(
            torch.Tensor([0.5**i for i in range(1, num_bits)])
        )

    def forward(self, inputs):
        bin = BinarizeFn.apply
        out = [inputs]
        for factor in self.weight:
            out.append(out[-1] - torch.abs(factor) * bin(out[-1]))
        return bin(torch.cat(out, dim=1))


class QuantizeTwoBit(torch.nn.Module):
    """
    Implementation of 2 bit residual quantization, the first bit is binarized the second uses  a factor similar to residual quantization
    Args:
      factors: the initial factor
    """

    def __init__(self, factors=0.5):
        super().__init__()
        self.factors = torch.nn.Parameter(torch.Tensor([factors]), requires_grad=True)

    def forward(self, input):
        binarize = BinarizeFn.apply
        first_half = binarize(input)
        second_half = input - self.factors * binarize(input)
        return binarize(torch.cat((first_half, second_half), dim=1))


def _init_quantizable_convolution(self, quantizer, bias, constraints):
    if isinstance(quantizer, Module):
        register_parametrization(self, "weight", quantizer)
        if bias:
            register_parametrization(self, "bias", quantizer)
    else:
        raise TypeError(f"Quantizer {quantizer} is not an instance of Module.")
    self.constraints = constraints if constraints else None

    def apply_constraint(self):
        if self.constraints:
            [constraint(self) for constraint in self.constraints]

    self.apply_constraint = types.MethodType(apply_constraint, self)


class QConv1d(torch.nn.Conv1d):
    """
    Implementation of quantized Conv1d layer,all parameters are equivalent to the base pytorch class except for the quantizer, and constraints.
    Args:
      quantizer: An instance of a quantizer for weight and bias , currently only 1 can be used for both
      constraints: A list of instances of constraints, applied with the apply constraint_call
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int],
        quantizer: Module,
        stride: tuple[int] = (1,),
        padding: int = 0,
        dilation: tuple[int] = (1,),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        constraints: list = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        _init_quantizable_convolution(
            self, quantizer=quantizer, bias=bias, constraints=constraints
        )


class QConv2d(torch.nn.Conv2d):
    """
    Implementation of quantized Conv2d layer, all parameters are equivalent to the base pytorch class except for the quantizer and consraints.
    Args:
     quantizer: An instance of a quantizer for weight and bias , currently only 1 can be used for both
    constraints: A list of instances of constraints, applied with the apply constraint_call
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        quantizer,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        constraints: list = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        _init_quantizable_convolution(
            self, quantizer=quantizer, bias=bias, constraints=constraints
        )


class ChannelShuffle(torch.nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        num_channels = x.data.size()[1]
        original_shape = x.data.size()[2:]
        x = x.view(
            x.data.size()[0], self.groups, num_channels // self.groups, *original_shape
        )
        x = x.transpose(1, 2).contiguous()
        return x.view(x.data.size()[0], num_channels, *original_shape)


class QLinear(torch.nn.Linear):
    """
    Implementation of quantized Linear layer,all parameters are equivalent to the base pytorch class except for the quantizer and constraints.
    Args:
     quantizer: An instance of a quantizer for weight and bias, currently only 1 can be used for both
     constraints: A list of instances of constraints, applied with the apply constraint_call
    """

    def __init__(
        self, in_features, out_features, quantizer, bias=True, constraints: list = None
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        _init_quantizable_convolution(
            self, quantizer=quantizer, bias=bias, constraints=constraints
        )


class QLSTMCell(torch.nn.LSTMCell):
    """
    Implementation of quantized LSTM cell,all parameters are equivalent to the base pytorch class except for the quantizers.
    Args:
        input_gate_activation:  The input gate activation, expects a quantizer instance, if None will default to sigmoid
        forget_gate_activation: The forget gate activation, expects a quantizer instance, if None will default to sigmoid
        cell_gate_activation: The cell gate activation, expects a quantizer instance, if None will default to tanh
        output_gate_activation: The output gate activation, expects a quantizer instance, if None will default to sigmoid
        new_cell_gate_activation: The new cell gate activation, expects a quantizer instance, if None will default to tanh
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_quantizer: Quantize,
        weight_quantizer: Quantize,
        bias: bool = True,
        input_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        forget_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        cell_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        output_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        new_cell_state_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
    ):
        super().__init__(input_size, hidden_size, bias)

        self.state_quantizer = state_quantizer
        self.weight_quantizer = weight_quantizer
        self.input_gate_activation = input_gate_activation
        self.forget_gate_activation = forget_gate_activation
        self.cell_gate_activation = cell_gate_activation
        self.output_gate_activation = output_gate_activation
        self.new_cell_state_activation = new_cell_state_activation

    @staticmethod
    def __identity(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor

    def forward(
        self, x: torch.Tensor, hx: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Implementation based on
        # https://github.com/pytorch/pytorch/blob/e9ef087d2d12051341db485c8ac64ea64649823d/benchmarks/fastrnns/cells.py#L25
        if hx is None:
            zeros = torch.zeros(
                x.size(0), self.hidden_size, dtype=x.dtype, device=x.device
            )
            hx = (zeros, zeros)

        h_0, c_0 = self.state_quantizer(hx[0]), self.state_quantizer(hx[1])
        weight_ih, weight_hh = self.weight_quantizer(
            self.weight_ih
        ), self.weight_quantizer(self.weight_hh)

        if self.bias:
            bias_ih, bias_hh = self.weight_quantizer(
                self.bias_ih
            ), self.weight_quantizer(self.bias_hh)
            gates = (
                torch.mm(x, weight_ih.t())
                + bias_ih
                + torch.mm(h_0, weight_hh.t())
                + bias_hh
            )
        else:
            gates = torch.mm(x, weight_ih.t()) + torch.mm(h_0, weight_hh.t())

        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(4, dim=1)

        i = self.input_gate_activation(in_gate)
        f = self.forget_gate_activation(forget_gate)
        g = self.cell_gate_activation(cell_gate)
        o = self.output_gate_activation(out_gate)

        c_1 = f * c_0 + i * g
        h_1 = o * self.new_cell_state_activation(c_1)

        return h_1, c_1


class QLSTM(torch.nn.Module):
    """
    Implementation of quantized LSTM
    Args:
     lstm_cell : an instance of a lstm_cell
     batch_first: same as pytorch. If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_quantizer: Quantize,
        weight_quantizer: Quantize,
        bias: bool = True,
        input_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        forget_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        cell_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        output_gate_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        new_cell_state_activation: Callable[[torch.Tensor], torch.Tensor] = torch.tanh,
        batch_first=False,
    ):
        super().__init__()
        self.cell = QLSTMCell(
            hidden_size=hidden_size,
            state_quantizer=state_quantizer,
            weight_quantizer=weight_quantizer,
            bias=bias,
            input_size=input_size,
            input_gate_activation=input_gate_activation,
            forget_gate_activation=forget_gate_activation,
            cell_gate_activation=cell_gate_activation,
            output_gate_activation=output_gate_activation,
            new_cell_state_activation=new_cell_state_activation,
        )
        self.batch_first = batch_first

    def forward(
        self, input: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if self.batch_first:
            input = torch.stack(torch.unbind(input), dim=1)

        # Implementation based on
        # https://github.com/pytorch/pytorch/blob/bb7fd1fcfbd2507272fd9b3f2610ef02bfba5692/benchmarks/fastrnns/custom_lstms.py#L184
        inputs = torch.unbind(input, dim=0)
        outputs = []
        for i in range(len(inputs)):
            output, cell_state = self.cell(inputs[i], state)
            state = output, cell_state
            outputs += [output]
        stack_dim = 1 if self.batch_first else 0
        result = torch.stack(outputs, dim=stack_dim)
        return result, state


class BatchNormedActivatedConv1d(torch.nn.Module):
    """Applies a convolution followed by a batchnorm and an activation function.
    The BatchNorm is not performing an affine translation, instead we incorporate
    a trainable scaling factor that is applied to each channel before the application
    of the activation function.
    """

    def __init__(
        self,
        activation: Callable[[], Quantize],
        kernel_size,
        in_channels,
        out_channels,
        groups,
        bias,
        channel_multiplexing_factor,
    ):
        super().__init__()
        self.conv = Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=groups,
            padding=0,
            bias=bias,
        )
        self.bn = BatchNorm1d(num_features=out_channels, affine=False)
        self.scaling_factors = Parameter(torch.ones((out_channels, 1)))
        self.quantize = activation()
        self.channel_multiplexing_factor = channel_multiplexing_factor

    @property
    def codomain_elements(self) -> list[float]:
        return self.quantize.codomain

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.scaling_factors * x
        x = self.quantize(x)
        return x


def define_batch_normed_convolution_1d(activation, channel_multiplexing_factor):
    """Allows to define a new `BatchNormedActivationConvolution` that uses `activation` for its activation function.
    The `channel_multiplexing_factor` will be used by calling modules to determine if the activation function will
    change the number of channels, like the `ResidualBinarization` does.
    This way we can determine the input shape for following layers."""

    def wrapper(cls):
        class Wrapped(BatchNormedActivatedConv1d):
            def __init__(self, kernel_size, in_channels, out_channels, groups, bias):
                super().__init__(
                    activation=activation,
                    kernel_size=kernel_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    bias=bias,
                    channel_multiplexing_factor=channel_multiplexing_factor,
                )

        return Wrapped

    return wrapper


@define_batch_normed_convolution_1d(Ternarize, 1)
class TernaryConvolution1d:
    pass


@define_batch_normed_convolution_1d(QuantizeTwoBit, 2)
class MultilevelResidualBinarizationConv1d:
    pass


@define_batch_normed_convolution_1d(Binarize, 1)
class BinaryActivatedConv1d:
    pass


class SplitConvolutionBase(torch.nn.Module):
    def __init__(
        self,
        convolution: TaggedModule,
        kernel_size,
        in_channels,
        out_channels,
        codomain_elements: list[float],
    ):
        """Use this as a base class if you want to incorporate your own convolution implementation (up to the quantizing activation function)
        into a SplitConvolution.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.codomain_elements = codomain_elements
        self.depthwise = convolution(
            kernel_size=kernel_size,
            in_channels=in_channels,
            groups=in_channels,
            out_channels=in_channels,
            bias=False,
        )  # type: ignore
        self.pointwise = convolution(
            kernel_size=1,
            in_channels=in_channels * self.depthwise.channel_multiplexing_factor,
            out_channels=out_channels,
            groups=1,
            bias=False,
        )
        depthwise_shape = (in_channels, kernel_size)

        def depthwise_input() -> Tensor:
            return create_codomain_for_depthwise_1d_conv(
                depthwise_shape, self.codomain_elements
            )

        depthwise_precomputable_tag = precomputable(
            input_shape=depthwise_shape, input_generator=depthwise_input
        )
        self.depthwise = depthwise_precomputable_tag(self.depthwise)

        pointwise_shape = (
            1,
            in_channels * self.depthwise.channel_multiplexing_factor,
        )

        def pointwise_input():
            return create_codomain_for_1d_conv(
                pointwise_shape, self.depthwise.codomain_elements
            )

        pointwise_precomputable_tag = precomputable(
            input_shape=pointwise_shape, input_generator=pointwise_input
        )
        self.pointwise = pointwise_precomputable_tag(self.pointwise)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


def define_split_convolution(
    conv_cls, _channel_multiplexing_factor, input_domain_elements
):
    def wrapper(cls):
        class Wrapped(SplitConvolutionBase):
            channel_multiplexing_factor = _channel_multiplexing_factor

            def __init__(self, kernel_size, in_channels, out_channels):
                super().__init__(
                    convolution=conv_cls,
                    kernel_size=kernel_size,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    codomain_elements=input_domain_elements,
                )

        return Wrapped

    return wrapper


@define_split_convolution(
    BinaryActivatedConv1d, _channel_multiplexing_factor=1, input_domain_elements=[-1, 1]
)
class BinarySplitConv:
    pass


@define_split_convolution(
    TernaryConvolution1d,
    _channel_multiplexing_factor=1,
    input_domain_elements=[-1, 0, 1],
)
class TernarySplitConv:
    pass


@define_split_convolution(
    MultilevelResidualBinarizationConv1d,
    _channel_multiplexing_factor=2,
    input_domain_elements=[-1, 1],
)
class TwoBitSplitConv:
    pass
