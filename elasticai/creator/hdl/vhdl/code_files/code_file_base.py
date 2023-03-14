class CodeFileBase:
    @property
    def name(self) -> str:
        return self._name

    def lines(self) -> list[str]:
        return self._code

    def __repr__(self) -> str:
        return f"CodeBaseFile(name={self._name}, code={self._code})"

    def __init__(self, name: str, code: list[str]):
        self._name = name
        self._code = code
