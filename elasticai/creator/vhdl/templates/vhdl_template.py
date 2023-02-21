from typing import Iterator

from elasticai.creator.resource_utils import read_text
from elasticai.creator.templates import AbstractBaseTemplate


class VHDLTemplate(AbstractBaseTemplate):
    def _read_raw_template(self) -> Iterator[str]:
        return read_text(
            self._template_package, f"{self._template_name}{self._template_file_suffix}"
        )

    _template_package = "elasticai.creator.vhdl.template_resources"
    _template_file_suffix = ".tpl.vhd"
    _generated_file_suffix = ".vhd"

    def __init__(self, base_name: str, **parameters: str | tuple[str] | list[str]):
        super().__init__(**parameters)
        self._template_name = base_name
        self._saved_raw_template: list[str] = []

    @property
    def name(self) -> str:
        return f"{self._template_name}{self._generated_file_suffix}"

    def code(self):
        yield from self.lines()
