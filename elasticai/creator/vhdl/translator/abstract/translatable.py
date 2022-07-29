from typing import Any, Protocol

from elasticai.creator.vhdl.vhdl_component import VHDLModule


class Translatable(Protocol):
    def translate(self, args: Any) -> VHDLModule:
        ...
