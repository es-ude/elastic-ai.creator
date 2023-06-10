from elasticai.creator.base_modules.sigmoid import Sigmoid

from .fp_monotonic_increasing_module import FPPrecomputedMonotonicIncreasingModule


class FPSigmoid(FPPrecomputedMonotonicIncreasingModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=Sigmoid(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
