import warnings
from dataclasses import dataclass

from elasticai.creator.vhdl.code.code_file import CodeFile
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import Tanh


@dataclass
class TanhComponent(CodeFile):
    x: list[FixedPoint]

    @property
    def name(self) -> str:
        return f"tanh.vhd"

    def lines(self) -> list[str]:
        return list(Tanh(x=self.x, component_name="tanh").code())

    def code(self) -> list[str]:
        warnings.warn(
            message=DeprecationWarning(
                (
                    f"calling instance directly is deprecated, use the"
                    f" lines() method instead "
                ),
            ),
            stacklevel=2,
        )
        return self.lines()
