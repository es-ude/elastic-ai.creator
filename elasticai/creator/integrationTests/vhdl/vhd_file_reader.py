from io import TextIOBase
from typing import Iterable, Iterator, TextIO


class VHDFileReaderWithoutComments:
    """
    Allows you to iterate over a text ignoring blank lines and vhdl comments.
    This is mainly used for integration tests
    """

    def __init__(self, file: TextIO):
        self._file = file

    @staticmethod
    def _line_is_relevant(line: str) -> bool:
        line_without_leading_whitespace = line.lstrip()
        return len(
            line_without_leading_whitespace
        ) > 0 and not line_without_leading_whitespace.startswith("--")

    @staticmethod
    def _strip_trailing_comment(line: str) -> str:
        return line.split(" --")[0]

    def __iter__(self) -> Iterator[str]:
        for line in self._file:
            if self._line_is_relevant(line):
                yield self._strip_trailing_comment(line)
