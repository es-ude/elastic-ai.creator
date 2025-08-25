from typing import Any, cast

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.base_modules.linear import Linear as LinearBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign
from elasticai.creator.nn.fixed_point.math_operations import MathOperations


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
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            operations=MathOperations(config=self._config),
            bias=bias,
            device=device,
        )

    def get_params(self) -> tuple[list[list[float]], list[float]]:
        bias = [0] * self.out_features if self.bias is None else self.bias.tolist()
        weights = self.weight.tolist()
        return weights, bias

    def get_params_quant(self) -> tuple[list[list[int]], list[int]]:
        weights, bias = self.get_params()
        q_weights = cast(list[list[int]], self._config.cut_as_integer(weights))
        q_bias = cast(list[int], self._config.cut_as_integer(bias))
        return q_weights, q_bias

    def create_design(self, name: str) -> LinearDesign:
        q_weights, q_bias = self.get_params_quant()
        return LinearDesign(
            frac_bits=self._config.frac_bits,
            total_bits=self._config.total_bits,
            in_feature_num=self.in_features,
            out_feature_num=self.out_features,
            weights=q_weights,
            bias=q_bias,
            name=name,
        )
