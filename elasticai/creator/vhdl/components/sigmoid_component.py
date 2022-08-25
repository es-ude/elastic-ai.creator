from dataclasses import dataclass

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Sigmoid


@dataclass
class SigmoidComponent:
    x: list[FixedPoint]

    @property
    def file_name(self) -> str:
        return f"sigmoid.vhd"

    def __call__(self) -> Code:
        yield from Sigmoid(x=self.x, component_name="sigmoid")()
