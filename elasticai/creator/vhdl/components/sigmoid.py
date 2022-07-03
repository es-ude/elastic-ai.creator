from typing import Optional

from elasticai.creator.resource_utils import PathType, read_text_from_path
from elasticai.creator.vhdl.language import Code
from elasticai.creator.vhdl.number_representations import FixedPoint
from elasticai.creator.vhdl.precomputed_scalar_function import (
    Sigmoid as PrecomputedSigmoid,
)
from elasticai.creator.vhdl.vhdl_component import VHDLComponent


class Sigmoid(VHDLComponent):
    def __init__(self, x: list[FixedPoint], component_name: str = "sigmoid") -> None:
        self._sigmoid = PrecomputedSigmoid(x=x, component_name=component_name)
        self._component_name = component_name

    @property
    def file_name(self) -> str:
        return f"{self._component_name}.vhd"

    def __call__(self, custom_template: Optional[PathType] = None) -> Code:
        if custom_template is None:
            yield from self._sigmoid()
        else:
            yield from read_text_from_path(custom_template).splitlines()
