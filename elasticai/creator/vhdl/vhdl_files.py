from elasticai.creator.resource_utils import read_text
from vhdl.code import Code, CodeFile


class VHDLFile(CodeFile):
    def save_to(self, prefix: str):
        raise NotImplementedError()

    _template_package = "elasticai.creator.vhdl.templates"

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return f"{self._name}.vhd"

    def code(self) -> Code:
        code = read_text(self._template_package, f"{self._name}.tpl.vhd")
        yield from code
