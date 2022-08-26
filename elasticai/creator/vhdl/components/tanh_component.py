from dataclasses import dataclass

from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Tanh


@dataclass
class TanhComponent:
    x: list[FixedPoint]

    @property
    def file_name(self) -> str:
        return f"tanh.vhd"

    def __call__(self) -> Code:
        yield from Tanh(x=self.x, component_name="tanh")()
