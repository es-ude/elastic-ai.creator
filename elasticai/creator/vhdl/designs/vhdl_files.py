from typing import Iterator

from ...resource_utils import read_text
from .template import TemplateImpl


class VHDLTemplate(TemplateImpl):
    def _read_raw_template(self) -> Iterator[str]:
        return read_text(
            self._template_package, f"{self._template_name}{self._template_file_suffix}"
        )

    _template_package = "elasticai.creator.vhdl.templates"
    _template_file_suffix = ".tpl.vhd"
    _generated_file_suffix = ".vhd"

    def __init__(self, template_name: str, **parameters: str | tuple[str] | list[str]):
        super().__init__(**parameters)
        self._template_name = template_name
        self._saved_raw_template: list[str] = []

    @property
    def name(self) -> str:
        return f"{self._template_name}{self._generated_file_suffix}"

    def code(self):
        yield from self.lines()
