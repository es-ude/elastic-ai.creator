from typing import Any

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.base_modules.prelu import PReLU as PReLUBase
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from .precomputed_module import PrecomputedModule


class PrecomputedPReLU(PrecomputedModule):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_steps: int,
        sampling_intervall: tuple[float, float] = (-float("inf"), float("inf")),
        num_parameters: int = 1,
        init: float = 0.25,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        """Quantized Activation Function for Leaky ReLU
        :param total_bits:          Total number of bits
        :param frac_bits:           Fraction of bits
        :param num_steps:           Number of LUT size / total steps in LUT inbetween the sampling interval
        :param sampling_intervall:  Floating tuple with input sampling interval (Note default is [-inf, inf] will be transformed into [-2.0, 1.75] for FxP(4, 2))
        """
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)

        super().__init__(
            base_module=PReLUBase(
                math_operations=MathOperations(self._config),
                num_parameters=num_parameters,
                init=init,
                device=device,
                dtype=dtype,
            ),
            total_bits=total_bits,
            frac_bits=frac_bits,
            num_steps=num_steps,
            sampling_intervall=sampling_intervall,
        )
