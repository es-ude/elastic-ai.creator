from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.file_generation.savable import Savable
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from .design import MacDesign


class MacLayer:
    def __init__(self, vector_width: int, fxp_params: FxpParams):
        self.ops = MathOperations(
            FxpArithmetic(
                FxpParams(
                    total_bits=fxp_params.total_bits, frac_bits=fxp_params.frac_bits
                )
            )
        )
        self._vector_width = vector_width
        self._fxp_params = fxp_params
        self.name = "fxp_mac"

    def __call__(self, a, b):
        return self.ops.matmul(a, b)

    def create_design(self, name: str) -> Savable:
        return MacDesign(
            fxp_params=self._fxp_params, vector_width=self._vector_width, name=name
        )
