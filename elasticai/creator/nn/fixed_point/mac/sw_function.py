from creator.file_generation.savable import Savable
from fixed_point._math_operations import MathOperations
from fixed_point._two_complement_fixed_point_config import FixedPointConfig
from fixed_point.mac.design import MacDesign


class MacLayer:
    def __init__(self, total_bits, frac_bits):
        self.ops = MathOperations(
            FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
        )
        self._total_bits = total_bits
        self._frac_bits = frac_bits

    def __call__(self, a, b):
        return self.ops.matmul(a, b)

    def create_design(self) -> Savable:
        return MacDesign(total_bits=self._total_bits, frac_bits=self._frac_bits)
