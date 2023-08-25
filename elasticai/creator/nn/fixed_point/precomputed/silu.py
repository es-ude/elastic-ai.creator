from fixed_point._math_operations import Operations
from fixed_point._two_complement_fixed_point_config import FixedPointConfig

from elasticai.creator.base_modules.siluwithtrainablescalebeta import (
    SiLUWithTrainableScaleBeta,
)

from .fp_precomputed_module import FPPrecomputedModule


class FPSiLU(FPPrecomputedModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=SiLUWithTrainableScaleBeta(
                arithmetics=Operations(
                    config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
                ),
            ),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
