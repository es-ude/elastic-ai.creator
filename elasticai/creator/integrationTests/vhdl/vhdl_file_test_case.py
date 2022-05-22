import importlib.resources
import unittest
from itertools import filterfalse

from elasticai.creator.vhdl.language import Code


class VHDLFileTestCase(unittest.TestCase):
    def compareToFile(self, vhdl_file: str, generated_code: Code):
        vhdl_code = importlib.resources.read_text(
            "elasticai.creator.integrationTests.vhdl", vhdl_file
        )

        def line_is_empty(line):
            return len(line) == 0

        vhdl_code = filterfalse(line_is_empty, map(str.strip, vhdl_code.splitlines()))

        def starts_with_comment(line):
            return line.startswith("--")

        vhdl_code = filterfalse(starts_with_comment, vhdl_code)
        vhdl_code = list(vhdl_code)
        generated_code = list(filterfalse(starts_with_comment, generated_code))
        self.assertEqual(vhdl_code, generated_code)
