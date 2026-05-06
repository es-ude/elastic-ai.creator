from typing import Any, cast

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.base_modules.prelu import PReLU as PReLUBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from .design import PReLU as PReLUDesign


class PReLU(DesignCreatorModule, PReLUBase):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_parameters: int = 1,
        init: float = 0.25,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        """Quantized Activation Function for Leaky ReLU
        :param total_bits:          Total number of bits
        :param frac_bits:           Number of fractional bits
        :param init:                Initial value of the negative slope
        """
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        super().__init__(
            math_operations=MathOperations(self._config),
            num_parameters=num_parameters,
            init=init,
            device=device,
            dtype=dtype,
        )
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._init = init

    def get_params(self) -> list[list[float]]:
        weights = self.weight.tolist()
        return weights

    def get_params_quant(self) -> list[list[float]]:
        weights = self.get_params()
        q_weights = cast(list[list[float]], self._config.cut_as_integer(weights))
        return q_weights

    def create_design(self, name: str) -> PReLUDesign:
        return PReLUDesign(
            name=name,
            total_bits=self._total_bits,
            frac_bits=self._frac_bits,
            weights=self.get_params_quant(),
        )
