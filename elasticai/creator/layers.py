import types
import warnings
from abc import abstractmethod
from typing import Optional, Tuple, Callable, Protocol, List, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from itertools import product
from torch.nn import Module
from torch.nn.utils.parametrize import register_parametrization

from elasticai.creator.functional import binarize as BinarizeFn

"""Implementation of quantizers and quantized variants of pytorch layers"""


class Quantize(Protocol):
    """
    Protocol to be adhered to by quantizations with properties to simplify translation to other levels of abstraction.
    Codomain is used for precalculation of convolutions by obtaining the codomain of the quantization prior to the convolution.
    Threshold is used on MVTUs.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        ...

    @property
    @abstractmethod
    def codomain(self) -> List[float]:
        ...

    @property
    @abstractmethod
    def thresholds(self) -> List[float]:
        ...


class Binarize(torch.nn.Module):
    """
    Implementation of binarizer with possible bits [-1,1]"""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return BinarizeFn.apply(x)

    @property
    def codomain(self):
        return [-1, 1]

    @property
    def thresholds(self):
        return torch.Tensor([0.0])

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

    @property
    def codomain(self):
        return [-1, 0, 1]

    @staticmethod
    def right_inverse(x: Tensor) -> Tensor:
        return x

    @property
    def thresholds(self):
        return torch.Tensor([0.5 / self.widening_factor, -0.5 / self.widening_factor])


class ResidualQuantization(Binarize):
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
            torch.Tensor([0.5 ** i for i in range(1, num_bits)])
        )

    def forward(self, inputs):
        bin = BinarizeFn.apply
        out = [inputs]
        for factor in self.weight:
            out.append(out[-1] - torch.abs(factor) * bin(out[-1]))
        return bin(torch.cat(out, dim=1))

    @property
    def codomain(self):
        return list(product([-1, 1], repeat=self.num_bits))

    @property
    def thresholds(self):
        threshold = [torch.Tensor([0.0])]
        for factor in self.weight:
            threshold.append(threshold[-1] + torch.abs(factor))

        return torch.flatten(torch.stack(threshold, 0))


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

    @property
    def codomain(self):
        return [[x, y] for x in [-1, 1] for y in [-1, 1]]

    @property
    def thresholds(self):
        return torch.cat((torch.Tensor([0.0]), self.factors), 0)


def _init_quantizable_convolution(self, quantizer, bias, constraints):
    warnings.warn(
        f"{type(self).__name__} is deprecated, use pytorch parametrization "
        "instead. "
        "See https://pytorch.org/tutorials/intermediate/parametrizations.html",
        DeprecationWarning,
        stacklevel=3,
    )
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
        kernel_size: Tuple[int, ...],
        quantizer: Module,
        stride: Tuple[int, ...] = (1,),
        padding: int = 0,
        dilation: Tuple[int, ...] = (1,),
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
        x = x.view(x.data.size()[0],self.groups,num_channels//self.groups,*original_shape)
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
        self, x: torch.Tensor, hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        self, input: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.batch_first:
            input = torch.stack(torch.unbind(input), dim=1)

        # Implementation based on
        # https://github.com/pytorch/pytorch/blob/master/benchmarks/fastrnns/custom_lstms.py#L184
        inputs = torch.unbind(input, dim=0)
        outputs = []
        for i in range(len(inputs)):
            state = output, cell_state = self.cell(inputs[i], state)
            outputs += [output]
        stack_dim = 1 if self.batch_first else 0
        outputs = torch.stack(outputs, dim=stack_dim)
        return outputs, state
