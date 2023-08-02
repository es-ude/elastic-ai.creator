from elasticai.creator.base_modules.arithmetics.fixed_point_arithmetics import (
    FixedPointArithmetics,
)
from elasticai.creator.base_modules.silu import SiLU
from elasticai.creator.base_modules.two_complement_fixed_point_config import (
    FixedPointConfig,
)

from .fp_monotonic_increasing_module import FPPrecomputedMonotonicIncreasingModule


class FPSiLU(FPPrecomputedMonotonicIncreasingModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=SiLU(
                arithmetics=FixedPointArithmetics(
                    config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
                ),
            ),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
