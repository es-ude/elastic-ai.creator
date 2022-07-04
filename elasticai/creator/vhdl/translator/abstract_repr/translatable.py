from typing import Any, Iterator, Protocol

from elasticai.creator.vhdl.translator.abstract_repr.custom_template_mapping import (
    CustomTemplateMapping,
)


class Translatable(Protocol):
    def translate(
        self, custom_template_mapping: CustomTemplateMapping, **kwargs: Any
    ) -> Iterator[tuple[str, list[str]]]:
        ...
