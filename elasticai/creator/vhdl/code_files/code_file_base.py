from elasticai.creator.vhdl.code.code import Code
from elasticai.creator.vhdl.code.code_file import CodeFile


class CodeFileBase(CodeFile):
    @property
    def name(self) -> str:
        return self._name

    def lines(self) -> Code:
        return self._code

    def __repr__(self) -> str:
        return f"CodeBaseFile(name={self._name}, code={self._code})"

    def __init__(self, name: str, code: Code):
        self._name = name
        self._code = code
