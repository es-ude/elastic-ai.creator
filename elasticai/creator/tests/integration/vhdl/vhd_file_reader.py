from typing import Iterator, TextIO


class VHDLFileReaderWithoutComments:
    """
    Allows you to iterate over a text ignoring blank lines and vhdl comments.
    This is mainly used for testing. That way we can compare expected and actually generated
    code without considering formatting and comments
    """

    def __init__(self, file: TextIO):
        self._file = file

    @staticmethod
    def _line_is_relevant(line: str) -> bool:
        return len(line) > 0 and not line.startswith("--")

    @staticmethod
    def _strip_trailing_comment(line: str) -> str:
        return line.split(" --")[0]

    def as_list(self) -> list[str]:
        return list(self)

    def __iter__(self) -> Iterator[str]:
        for line in self._file:
            line = line.rstrip("\n")
            line = line.strip()
            if self._line_is_relevant(line):
                line = self._strip_trailing_comment(line)
                yield line
