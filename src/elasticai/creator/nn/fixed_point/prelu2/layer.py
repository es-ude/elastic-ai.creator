from typing import Any

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.base_modules.prelu2 import PReLU2 as PReLU2Base
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point.math_operations import MathOperations

from .design import PReLU2 as PReLU2Design


class PReLU2(DesignCreatorModule, PReLU2Base):
    def __init__(
        self,
        total_bits: int,
        frac_bits: int,
        num_parameters: int = 1,
        init: float = 0.25,
        device: Any = None,
        dtype: Any = None,
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
            num_parameters=num_parameters,
            init=init,
            device=device,
            dtype=dtype,
        )
        self._total_bits = total_bits
        self._frac_bits = frac_bits
        self._init = init

    def get_params(self) -> list[list[float]]:
        return self._get_weight_exponent().tolist()

    def get_params_quant(self) -> list[list[int]]:
        weights = self.get_params()
        q_weights = [(-1) * int(val) - 1 for val in weights]
        return q_weights

    def create_design(self, name: str) -> PReLU2Design:
        return PReLU2Design(
            name=name,
            total_bits=self._total_bits,
            frac_bits=self._frac_bits,
            weights=self.get_params_quant(),
        )
