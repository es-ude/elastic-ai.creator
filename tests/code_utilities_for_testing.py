import unittest
from io import StringIO
from typing import Iterable, Iterator, TextIO, Union, cast

from elasticai.creator.resource_utils import get_file_from_package


class VHDLCodeTestCase(unittest.TestCase):
    def __init__(self, method_name="runTest") -> None:
        super().__init__(method_name)
        self.expected_code: list[str] = []

    def read_expected_code_from_file(self, file_name: str):
        with get_file_from_package(
            "elasticai.creator.tests.integration.vhdl", file_name
        ) as file:
            with open(file, "r") as opened_file:
                self.expected_code = VHDLReaderWithoutComments(
                    cast(
                        TextIO, opened_file
                    )  # for some reason mypy detects the variable as str, so we cast
                ).as_list()

    @staticmethod
    def unified_vhdl_from_module(module):
        vhdl_file = next(iter(module.files))
        code = "\n".join(vhdl_file.lines())
        io = StringIO(code)
        codes = VHDLReaderWithoutComments(io).as_list()
        return codes

    @staticmethod
    def code_section_from_string(s: str) -> list[list[str]]:
        return [VHDLCodeTestCase.code_from_string(s)]

    @staticmethod
    def code_from_string(s: str) -> list[str]:
        return VHDLReaderWithoutComments(StringIO(s))

    def check_contains_all_expected_lines(self, expected: list[str], actual: list[str]):
        reusable_code = list(actual)
        for line in expected:
            with self.subTest(line):
                self.assertTrue(
                    line in reusable_code,
                    f"expected to find: {line}\nbut found: {reusable_code}",
                )

    def check_lines_are_equal_ignoring_order(
        self, expected: Iterable[list[str]], actual: Iterable[list[str]]
    ):
        def unpack(codes: Iterable[list[str]]) -> list[list[str]]:
            return list(map(lambda x: list(x), codes))

        def sort(codes: Iterable[list[str]]) -> Iterable[list[str]]:
            return sorted(map(lambda x: sorted(x), codes))

        actual = unpack(sort(actual))
        expected = unpack(sort(expected))
        self.assertEqual(expected, actual)

    @staticmethod
    def extract_section_from_code(
        begin: Union[str, list[str]], end: Union[str, list[str]], lines: list[str]
    ) -> list[list[str]]:
        extract = False
        content: list[list[str]] = []
        current_section: list[str] = []
        begin = list(begin) if not isinstance(begin, str) else [begin]
        end = list(end) if not isinstance(end, str) else [end]
        look_back = len(begin)
        look_ahead = len(end)
        lines = list(lines)
        i = 0
        last_i = len(lines)
        while i < last_i:
            look_ahead_window = lines[i : i + look_ahead]
            look_back_window = lines[i : i + look_back]

            if not extract and look_back_window == begin:
                extract = True
                i = i + look_back - 1
            elif extract and look_ahead_window == end:
                extract = False
                content.append(current_section)
                current_section = []
            elif extract:
                current_section.append(lines[i])
            i += 1

        if extract:
            raise ValueError(f"reached end of code before end: {end}")
        return content


class VHDLReaderWithoutComments:
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
