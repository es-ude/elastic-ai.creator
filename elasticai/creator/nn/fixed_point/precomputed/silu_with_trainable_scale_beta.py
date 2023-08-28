from elasticai.creator.base_modules.silu_with_trainable_scale_beta import (
    SiLUWithTrainableScaleBeta as SiLUWithTrainableScaleBetaBase,
)
from elasticai.creator.nn.fixed_point._math_operations import MathOperations
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)

from .precomputed_module import PrecomputedModule


class SiLUWithTrainableScaleBeta(PrecomputedModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=SiLUWithTrainableScaleBetaBase(
                operations=MathOperations(
                    config=FixedPointConfig(total_bits=total_bits, frac_bits=frac_bits)
                ),
            ),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
