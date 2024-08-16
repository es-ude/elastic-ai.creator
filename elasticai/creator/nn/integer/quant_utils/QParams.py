import torch
from torch import nn

from .CalculateScaleZeropoint import calculateScaleZeropoint
from .DeQuantizeProcess import dequantizeProcess
from .QuantizeProcess import quantizeProcess


class QParams(nn.Module):
    def __init__(self, is_symmetric: bool, quant_bits: int, observer: nn.Module):
        super().__init__()
        self.observer = observer

        self.register_buffer(
            "min_quant", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "max_quant", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )

        if is_symmetric == False:
            self.min_quant.copy_(torch.tensor((-(1 << (quant_bits - 1)))))  # -128
        else:
            self.min_quant.copy_(torch.tensor((-(1 << (quant_bits - 1))) + 1))  # -127
        self.max_quant.copy_(torch.tensor(((1 << (quant_bits - 1)) - 1)))  # 127

        self.register_buffer(
            "scale", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "zero_point", torch.zeros((1), dtype=torch.int32, requires_grad=False)
        )
        self.register_buffer(
            "min_float", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "max_float", torch.ones((1), dtype=torch.float32, requires_grad=False)
        )
        self.register_buffer(
            "eps",
            torch.tensor(
                (torch.finfo(torch.float32).eps),
                dtype=torch.float32,
                requires_grad=False,
            ),
        )

        self.is_symmetric = is_symmetric
        self.quant_bits = quant_bits

    def updateScaleZeropoint(self, x_r: float) -> None:
        self.observer(x_r)
        scale, zero_point, min_float, max_float = calculateScaleZeropoint(
            is_symmetric=self.is_symmetric,
            min_quant=self.min_quant,
            max_quant=self.max_quant,
            min_float=self.observer.min_float,
            max_float=self.observer.max_float,
            eps=self.eps,
        )
        self.scale.copy_(scale)
        self.zero_point.copy_(zero_point)
        self.min_float.copy_(min_float)
        self.max_float.copy_(max_float)

    def set_scale(self, given_scale: float) -> None:
        self.scale.copy_(given_scale)

    def set_zero_point(self, given_zero_point: int) -> None:
        self.zero_point.copy_(given_zero_point)

    def set_quant_range(self, given_min_quant: int, given_max_quant: int) -> None:
        self.min_quant.copy_(given_min_quant)
        self.max_quant.copy_(given_max_quant)

    def quantizeProcess(self, x: torch.FloatTensor) -> torch.IntTensor:
        return quantizeProcess(
            x=x,
            min_quant=self.min_quant,
            max_quant=self.max_quant,
            scale_factor=self.scale,
            zero_point=self.zero_point,
        )

    def dequantizeProcess(self, x_q: int) -> float:
        return dequantizeProcess(x_q, self.scale, self.zero_point, self.quant_bits + 1)

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
            "scale",
            "zero_point",
            "min_float",
            "max_float",
            "min_quant",
            "max_quant",
        ]
        for key in key_names:
            value = getattr(self, key)
            value.data = state_dict[prefix + key].data
            state_dict.pop(prefix + key)

    def __str__(self):
        info = "scale: %.10f " % self.scale
        info += "zero_point: %d " % self.zero_point
        info += "min_float: %.6f " % self.min_float
        info += "max_float: %.6f" % self.max_float
        info += "min_quant: %d " % self.min_quant
        info += "max_quant: %d" % self.max_quant
        return info
