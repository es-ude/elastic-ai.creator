from typing import Callable, Optional

import torch
from torch.nn.utils.parametrize import register_parametrization

from elasticai.creator.base_modules.functional import binarize as BinarizeFn
from elasticai.creator.mlframework import Module, Tensor

"""Implementation of quantizers and quantized variants of pytorch layers"""


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


class Identity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
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


def _hook_quantizer(module: torch.nn.Module, quantizer: Module, bias: bool):
    if isinstance(quantizer, Module):
        register_parametrization(module, "weight", quantizer)
        if bias:
            register_parametrization(module, "bias", quantizer)
    else:
        raise TypeError(f"Quantizer {quantizer} is not an instance of Module.")


class QConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        quantizer: Module,
        stride: tuple[int] = (1,),
        padding: int = 0,
        dilation: tuple[int] = (1,),
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
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

        _hook_quantizer(self, quantizer=quantizer, bias=bias)


class QConv2d(torch.nn.Conv2d):
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
        _hook_quantizer(self, quantizer=quantizer, bias=bias)


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
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantizer: Module,
        bias: bool = True,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        _hook_quantizer(self, quantizer=quantizer, bias=bias)


class _QLSTMCellBase(torch.nn.LSTMCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_quantizer: Module,
        weight_quantizer: Module,
        bias: bool,
        input_gate_activation: Callable[[torch.Tensor], torch.Tensor],
        forget_gate_activation: Callable[[torch.Tensor], torch.Tensor],
        cell_gate_activation: Callable[[torch.Tensor], torch.Tensor],
        output_gate_activation: Callable[[torch.Tensor], torch.Tensor],
        new_cell_state_activation: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__(input_size, hidden_size, bias)

        self.state_quantizer = state_quantizer
        self.weight_quantizer = weight_quantizer
        self.input_gate_activation = input_gate_activation
        self.forget_gate_activation = forget_gate_activation
        self.cell_gate_activation = cell_gate_activation
        self.output_gate_activation = output_gate_activation
        self.new_cell_state_activation = new_cell_state_activation

    def forward(
        self, x: torch.Tensor, state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batched = x.dim() == 2
        if state is None:
            zeros_shape = (
                (x.size(dim=0), self.hidden_size) if batched else (self.hidden_size,)
            )
            zeros = torch.zeros(*zeros_shape, dtype=x.dtype, device=x.device)
            state = (zeros, zeros)

        h_0 = self.state_quantizer(state[0])
        c_0 = self.state_quantizer(state[1])
        weight_ih = self.weight_quantizer(self.weight_ih)
        weight_hh = self.weight_quantizer(self.weight_hh)

        gates = torch.matmul(x, weight_ih.t()) + torch.matmul(h_0, weight_hh.t())

        if self.bias:
            bias_ih = self.weight_quantizer(self.bias_ih)
            bias_hh = self.weight_quantizer(self.bias_hh)
            gates += bias_ih + bias_hh

        in_gate, forget_gate, cell_gate, out_gate = gates.chunk(
            4, dim=1 if batched else 0
        )

        i = self.input_gate_activation(in_gate)
        f = self.forget_gate_activation(forget_gate)
        g = self.cell_gate_activation(cell_gate)
        o = self.output_gate_activation(out_gate)

        c_1 = f * c_0 + i * g
        h_1 = o * self.new_cell_state_activation(c_1)

        return h_1, c_1


class _QLSTMBase(torch.nn.Module):
    def __init__(self, lstm_cell: _QLSTMCellBase, batch_first: bool):
        super().__init__()
        self.cell = lstm_cell
        self.batch_first = batch_first

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batched = x.dim() == 3

        if batched and self.batch_first:
            x = torch.stack(torch.unbind(x), dim=1)

        if state is not None:
            state = state[0].squeeze(0), state[1].squeeze(0)

        inputs = torch.unbind(x, dim=0)

        outputs = []
        for i in range(len(inputs)):
            hidden_state, cell_state = self.cell(inputs[i], state)
            state = (hidden_state, cell_state)
            outputs.append(hidden_state)

        result = torch.stack(outputs, dim=1 if batched and self.batch_first else 0)
        hidden_state, cell_state = state[0].unsqueeze(0), state[1].unsqueeze(0)

        return result, (hidden_state, cell_state)


class QLSTMCell(_QLSTMCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_quantizer: Module,
        weight_quantizer: Module,
        bias: bool = True,
    ) -> None:
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            state_quantizer=state_quantizer,
            weight_quantizer=weight_quantizer,
            bias=bias,
            input_gate_activation=torch.sigmoid,
            forget_gate_activation=torch.sigmoid,
            cell_gate_activation=torch.tanh,
            output_gate_activation=torch.sigmoid,
            new_cell_state_activation=torch.tanh,
        )


class QLSTM(_QLSTMBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_quantizer: Module,
        weight_quantizer: Module,
        bias: bool = True,
        batch_first: bool = False,
    ) -> None:
        super().__init__(
            lstm_cell=QLSTMCell(
                input_size=input_size,
                hidden_size=hidden_size,
                state_quantizer=state_quantizer,
                weight_quantizer=weight_quantizer,
                bias=bias,
            ),
            batch_first=batch_first,
        )
