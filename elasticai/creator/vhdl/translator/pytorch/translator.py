from typing import Iterator

from torch.nn import Module

from elasticai.creator.vhdl.vhdl_component import VHDLModule


class Translator:
    def translate(self, model: Module) -> Iterator[VHDLModule]:
        ...
