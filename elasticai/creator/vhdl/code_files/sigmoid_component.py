from dataclasses import dataclass
from typing import Iterator

from elasticai.creator.vhdl.code import CodeFile
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Sigmoid


@dataclass
class SigmoidComponent(CodeFile):
    x: list[FixedPoint]

    @property
    def name(self) -> str:
        return f"sigmoid.vhd"

    def __iter__(self) -> Iterator[str]:
        yield from Sigmoid(x=self.x, component_name="sigmoid").code()

    def code(self) -> Code:
        return self
