import torch
import torch.nn.functional as f

from elasticai.creator.nn.autograd_functions.fixed_point_quantization import (
    FixedPointDequantFunction,
    FixedPointQuantFunction,
)
from elasticai.creator.nn.quant_typings import QuantType
from elasticai.creator.vhdl.number_representations import FixedPointFactory


class _HardTanhBase(torch.nn.Hardtanh):
    def __init__(
        self,
        min_val: float = -1,
        max_val: float = 1,
        inplace: bool = False,
        input_quant: QuantType = lambda x: x,
        input_dequant: QuantType = lambda x: x,
    ) -> None:
        super().__init__(min_val=min_val, max_val=max_val, inplace=inplace)
        self.input_quant = input_quant
        self.input_dequant = input_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return f.hardtanh(
            self.input_dequant(self.input_quant(x)),
            min_val=self.min_val,
            max_val=self.max_val,
            inplace=self.inplace,
        )

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("The quantized_forward function is not implemented.")


class FixedPointHardTanh(_HardTanhBase):
    def __init__(
        self,
        fixed_point_factory: FixedPointFactory,
        min_val: float = -1,
        max_val: float = 1,
        inplace: bool = False,
    ) -> None:
        super().__init__(
            min_val=min_val,
            max_val=max_val,
            inplace=inplace,
            input_quant=lambda x: FixedPointQuantFunction.apply(x, fixed_point_factory),
            input_dequant=lambda x: FixedPointDequantFunction.apply(
                x, fixed_point_factory
            ),
        )
        self.fp_factory = fixed_point_factory

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        def fp(value: float) -> int:
            largest_fp_int = 2 ** (self.fp_factory.total_bits - 1) - 1
            int_value = int(value * (1 << self.fp_factory.frac_bits))
            return min(max(int_value, -largest_fp_int), largest_fp_int)

        return f.hardtanh(x, min_val=fp(self.min_val), max_val=fp(self.max_val))
