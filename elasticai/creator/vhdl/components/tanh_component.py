from dataclasses import dataclass

from vhdl.code import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Tanh


@dataclass
class TanhComponent:
    x: list[FixedPoint]

    @property
    def name(self) -> str:
        return f"tanh.vhd"

    @property
    def code(self) -> Code:
        yield from Tanh(x=self.x, component_name="tanh")()
