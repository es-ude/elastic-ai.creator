import unittest
from io import StringIO
from typing import Iterable, Union

from elasticai.creator.tests.integration.vhdl.vhd_file_reader import (
    VHDLFileReaderWithoutComments,
)
from elasticai.creator.vhdl.code import Code, CodeModule


class CodeTestCase(unittest.TestCase):
    @staticmethod
    def unified_vhdl_from_module(module: CodeModule):
        vhdl_file = next(iter(module.files))
        code = "\n".join(vhdl_file.code())
        io = StringIO(code)
        codes = VHDLFileReaderWithoutComments(io).as_list()
        return codes

    @staticmethod
    def code_section_from_string(s: str) -> list[Code]:
        return [CodeTestCase.code_from_string(s)]

    @staticmethod
    def code_from_string(s: str) -> Code:
        return VHDLFileReaderWithoutComments(StringIO(s))

    def check_contains_all_expected_lines(self, expected: Code, actual: Code):
        reusable_code = list(actual)
        for line in expected:
            with self.subTest(line):
                self.assertTrue(
                    line in reusable_code,
                    f"expected to find: {line}\nbut found: {reusable_code}",
                )

    def check_lines_are_equal_ignoring_order(
        self, expected: Iterable[Code], actual: Iterable[Code]
    ):
        def unpack(codes: Iterable[Code]) -> list[list[str]]:
            return list(map(lambda x: list(x), codes))

        def sort(codes: Iterable[Code]) -> Iterable[Code]:
            return sorted(map(lambda x: sorted(x), codes))

        actual = unpack(sort(actual))
        expected = unpack(sort(expected))
        self.assertEqual(expected, actual)

    @staticmethod
    def extract_section_from_code(
        begin: Union[str, Code], end: Union[str, Code], lines: Code
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
