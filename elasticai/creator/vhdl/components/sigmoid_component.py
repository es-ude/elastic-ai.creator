from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import (
    Sigmoid as PrecomputedSigmoid,
)


class SigmoidComponent:
    def __init__(self, x: list[FixedPoint], component_name: str = "sigmoid") -> None:
        self._sigmoid = PrecomputedSigmoid(x=x, component_name=component_name)
        self._component_name = component_name

    @property
    def file_name(self) -> str:
        return f"{self._component_name}.vhd"

    def __call__(self) -> Code:
        yield from self._sigmoid()
