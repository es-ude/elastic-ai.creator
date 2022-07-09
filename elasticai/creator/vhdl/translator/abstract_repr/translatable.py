from typing import Any, Iterator, Protocol

from elasticai.creator.vhdl.language import Code


class Translatable(Protocol):
    def translate(self, *args: Any, **kwargs: Any) -> Iterator[tuple[str, Code]]:
        ...
