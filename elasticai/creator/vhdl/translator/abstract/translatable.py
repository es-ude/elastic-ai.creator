from typing import Any, Iterator, Protocol

from elasticai.creator.vhdl.vhdl_component import VHDLComponent


class Translatable(Protocol):
    def translate(self, *args: Any, **kwargs: Any) -> Iterator[VHDLComponent]:
        ...
