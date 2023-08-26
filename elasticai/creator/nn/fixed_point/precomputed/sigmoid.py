from elasticai.creator.base_modules.sigmoid import Sigmoid as SigmoidBase

from .precomputed_module import PrecomputedModule


class Sigmoid(PrecomputedModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-10, 10),
    ) -> None:
        super().__init__(
            base_module=SigmoidBase(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
