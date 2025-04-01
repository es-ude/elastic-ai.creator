from typing import Any, cast
import torch

from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign
from elasticai.creator.nn.fixed_point.linear.testbench import LinearTestbench
from elasticai.creator.nn.fixed_point.math_operations import MathOperations
from elasticai.creator.nn.fixed_point.two_complement_fixed_point_config import (
    FixedPointConfig,
)


class Linear(DesignCreatorModule, LinearBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bits: int,
        frac_bits: int,
        bias: bool = True,
        device: Any = None,
        parallel : bool = False,
    ) -> None:
        self._config = FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        self._double_config = FixedPointConfig(total_bits=2*total_bits, frac_bits=2*frac_bits)
        self.parallel = parallel
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperations(config=self._config),
            double_operations=MathOperations(config=self._double_config),
            bias=bias,
            device=device,
        )

    def create_design(self, name: str) -> LinearDesign:
        def float_to_signed_int(value: float | list) -> int | list:
            if isinstance(value, list):
                return list(map(float_to_signed_int, value))
            return self._config.as_integer(value)

        bias = [0] * self.out_features if self.bias is None else self._operations.quantize(self.bias).tolist()
        signed_int_weights = cast(
            list[list[int]], float_to_signed_int(self._operations.quantize(self.weight).tolist())
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
            parallel=self.parallel
        )

    def create_testbench(self, name: str, uut: LinearDesign) -> LinearTestbench:
        return LinearTestbench(name, uut)
