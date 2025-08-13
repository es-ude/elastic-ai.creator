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
        sampling_intervall: tuple[float, float] = (-float("inf"), float("inf")),
    ) -> None:
        """Quantized Activation Function for Adaptive SiLU
        :param total_bits:          Total number of bits
        :param frac_bits:           Fraction of bits
        :param num_steps:           Number of LUT size / total steps in LUT inbetween the sampling interval
        :param sampling_intervall:  Floating tuple with input sampling interval (Note default is [-inf, inf] will be transformed into [-2.0, 1.75] for FxP(4, 2))
        """
        super().__init__(
            base_module=AdaptableSiLUBase(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
