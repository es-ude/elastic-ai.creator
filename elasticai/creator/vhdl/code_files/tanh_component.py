from dataclasses import dataclass
from typing import Iterator

from elasticai.creator.vhdl.code import CodeFile
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Tanh


@dataclass
class TanhComponent(CodeFile):
    x: list[FixedPoint]

    @property
    def name(self) -> str:
        return f"tanh.vhd"

    def __iter__(self) -> Iterator[str]:
        yield from Tanh(x=self.x, component_name="tanh").code()

    def code(self) -> Code:
        return self
