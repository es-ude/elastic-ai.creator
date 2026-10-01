from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.base_modules.leaky_relu2 import LeakyReLU2 as LeakyReLU2Base
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from .design import LeakyReLU2 as LeakyReLU2Design


class LeakyReLU2(DesignCreatorModule, LeakyReLU2Base):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        init: float = 0.25,
    ) -> None:
        """Quantized Activation Function for Leaky ReLU (only power of 2 scaling values are supported)
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
            init=init,
        )
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._init = init

    def get_params(self) -> float:
        return float(self._get_weight_exponent())

    def get_params_quant(self) -> list[int]:
        weights = self.get_params()
        return [(-1) * int(weights) - 1]

    def create_design(self, name: str) -> LeakyReLU2Design:
        return LeakyReLU2Design(
            name=name,
            total_bits=self._total_bits,
            frac_bits=self._frac_bits,
            weights=self.get_params_quant(),
        )
