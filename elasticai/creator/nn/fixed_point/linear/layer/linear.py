from typing import Any, cast

from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign
from elasticai.creator.nn.fixed_point.linear.testbench import LinearTestbench


class Linear(DesignCreatorModule, LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bits: int,
        frac_bits: int,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperations(config=self._config),
            bias=bias,
            device=device,
        )

    def create_design(self, name: str) -> LinearDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.as_integer(value)

        bias = [0] * self.out_features if self.bias is None else self.bias.tolist()
        signed_int_weights = cast(
            list[list[int]], float_to_signed_int(self.weight.tolist())
        )
        signed_int_bias = cast(list[int], float_to_signed_int(bias))

        return LinearDesign(
            frac_bits=self._config.frac_bits,
            total_bits=self._config.total_bits,
            in_feature_num=self.in_features,
            out_feature_num=self.out_features,
            weights=signed_int_weights,
            bias=signed_int_bias,
            name=name,
        )

    def create_testbench(self, name: str, uut: LinearDesign) -> LinearTestbench:
        return LinearTestbench(name, uut)
