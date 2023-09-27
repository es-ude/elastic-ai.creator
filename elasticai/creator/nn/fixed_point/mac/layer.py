from elasticai.creator.file_generation.savable import Savable

from .._math_operations import MathOperations
from .._two_complement_fixed_point_config import FixedPointConfig
from .design import MacDesign
from .mactestbench import MacTestBench
from .number_converter import FXPParams


class MacLayer:
    def __init__(self, vector_width: int, fxp_params: FXPParams):
        self.ops = MathOperations(
            FixedPointConfig(
                total_bits=fxp_params.total_bits, frac_bits=fxp_params.frac_bits
            )
        )
        self._vector_width = vector_width
        self._fxp_params = fxp_params

    def __call__(self, a, b):
        return self.ops.matmul(a, b)

    def create_design(self, name: str) -> Savable:
        return MacDesign(
            fxp_params=self._fxp_params, vector_width=self._vector_width, name=name
        )

    def create_testbench(self, name: str) -> MacTestBench:
        return MacTestBench(
            fxp_params=self._fxp_params,
            uut=self.create_design("mac_wrapper_test"),
            uut_name="mac_wrapper_test",
            name=name,
        )
