from elasticai.creator.base_modules.adaptable_silu import (
    AdaptableSiLU as AdaptableSiLUBase,
)

from .precomputed_module import PrecomputedModule


class AdaptableSiLU(PrecomputedModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=AdaptableSiLUBase(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
