import warnings
from dataclasses import dataclass

from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Sigmoid


@dataclass
class SigmoidComponent:
    x: list[FixedPoint]

    @property
    def name(self) -> str:
        return f"sigmoid.vhd"

    def lines(self) -> list[str]:
        return list(Sigmoid(x=self.x, component_name="sigmoid").code())

    def code(self) -> list[str]:
        warnings.warn(
            message=DeprecationWarning(
                f"call is deprecated, use the lines() method instead ",
            ),
            stacklevel=2,
        )
        return self.lines()
