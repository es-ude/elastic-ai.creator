import torch
import torch.nn as nn

# from elasticai.creator.nn.integer.quant_utils.dequantize import dequantizeProcess
# from elasticai.creator.nn.integer.quant_utils.quantize import quantizeProcess
from elasticai.creator.nn.integer.math_operations.subtraction import subtract
from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_quant_params,
)


class QParams(nn.Module):
    def __init__(
        self, quant_bits: int, observer: nn.Module, is_symmetric: bool, is_signed: bool
    ):
        super().__init__()

        self.observer = observer
        self.quant_bits = quant_bits
        self.is_symmetric = is_symmetric
        self.is_signed = is_signed

        self.register_buffer(
            "min_float", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "max_float", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "min_quant", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "max_quant", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "scale_factor", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "zero_point", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "eps",
            torch.tensor(
                (torch.finfo(torch.float32).eps),
                dtype=torch.float32,
                requires_grad=False,
            ),
        )

        self._calculate_quant_range()

    def _calculate_quant_range(self):
        if self.is_signed:
            min_value = -(1 << (self.quant_bits - 1)) + (
                1 if self.is_symmetric else 0
            )  # -127 or -128 for 8 bits
            self.min_quant.copy_(torch.tensor(min_value))

            self.max_quant.copy_(
                torch.tensor((1 << (self.quant_bits - 1)) - 1)  # 127 for 8 bits
            )
        else:
            self.min_quant.copy_(torch.tensor(0))  # 0 for 8 bits

            max_value = (1 << self.quant_bits) - (
                2 if self.is_symmetric else 1
            )  # 254 or 255 for 8 bits
            self.max_quant.copy_(torch.tensor(max_value))

    def update_quant_params(self, x: torch.FloatTensor) -> None:
        self.observer(x)
        scale_factor, zero_point, min_float, max_float = calculate_quant_params(
            is_symmetric=self.is_symmetric,
            min_quant=self.min_quant,
            max_quant=self.max_quant,
            min_float=self.observer.min_float,
            max_float=self.observer.max_float,
            eps=self.eps,
        )

        for attr, value in zip(
            [self.scale_factor, self.zero_point, self.min_float, self.max_float],
            [scale_factor, zero_point, min_float, max_float],
        ):
            attr.copy_(value)

    def set_scale_factor(self, given_scale_factor: torch.FloatTensor) -> None:
        self.scale_factor.copy_(given_scale_factor)

    def set_zero_point(self, given_zero_point: torch.IntTensor) -> None:
        self.zero_point.copy_(given_zero_point)

    def set_quant_range(self, given_quant_bits: int) -> None:
        self.quant_bits = given_quant_bits
        self._calculate_quant_range()

    def quantize(self, x: torch.FloatTensor) -> torch.IntTensor:
        x_q = x / self.scale_factor + self.zero_point
        x_q = x_q.round_().clamp(min=self.min_quant.item(), max=self.max_quant.item())
        x_q = x_q.to(torch.int32)
        return x_q

    def dequantize(self, x_q: torch.IntTensor) -> torch.FloatTensor:
        x_q = subtract(x_q, self.zero_point, self.quant_bits + 1)
        x_dq = x_q.to(torch.float32) * self.scale_factor
        return x_dq

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_names = [
            "min_float",
            "max_float",
            "min_quant",
            "max_quant",
            "scale_factor",
            "zero_point",
        ]
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info += "min_float: %.6f " % self.min_float
        info += "max_float: %.6f" % self.max_float
        info += "min_quant: %d " % self.min_quant
        info += "max_quant: %d" % self.max_quant
        info += "scale_factor: %.10f " % self.scale_factor
        info += "zero_point: %d " % self.zero_point
        return info


class AsymmetricSignedQParams(QParams):
    def __init__(self, quant_bits: int, observer: nn.Module):
        super().__init__(quant_bits, observer, is_symmetric=False, is_signed=True)


class SymmetricSignedQParams(QParams):
    def __init__(self, quant_bits: int, observer: nn.Module):
        super().__init__(quant_bits, observer, is_symmetric=True, is_signed=True)


class AsymmetricUnsignedQparams(QParams):
    def __init__(self, quant_bits: int, observer: nn.Module):
        super().__init__(quant_bits, observer, is_symmetric=False, is_signed=False)


class SymmetricUnsignedQParams(QParams):
    def __init__(self, quant_bits: int, observer: nn.Module):
        super().__init__(quant_bits, observer, is_symmetric=True, is_signed=False)
