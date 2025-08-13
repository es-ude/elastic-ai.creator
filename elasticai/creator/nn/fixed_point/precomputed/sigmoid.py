from elasticai.creator.base_modules.sigmoid import Sigmoid as SigmoidBase

from .precomputed_module import PrecomputedModule


class Sigmoid(PrecomputedModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-float("inf"), float("inf")),
    ) -> None:
        """Quantized Activation Function for Sigmoid
        :param total_bits:          Total number of bits
        :param frac_bits:           Fraction of bits
        :param num_steps:           Number of LUT size / total steps of LUT inbetween sampling intervall
        :param sampling_intervall:  Floating tuple with input sampling interval (Note default is [-inf, inf] will be transformed into [-2.0, 1.75] for FxP(4, 2))
        """
        super().__init__(
            base_module=SigmoidBase(),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
