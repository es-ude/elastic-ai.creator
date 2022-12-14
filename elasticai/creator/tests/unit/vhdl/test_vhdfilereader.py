import unittest
from io import StringIO

from elasticai.creator.tests.code_utilities_for_testing import VHDLReaderWithoutComments


class VHDFileReaderWithoutCommentsTest(unittest.TestCase):
    def check(self, input: str, expected: str):
        io_dummy = StringIO(input)
        reader = VHDLReaderWithoutComments(io_dummy)
        expected_code = tuple(expected.splitlines())
        actual = tuple(reader)
        self.assertEqual(expected_code, actual)

    def test_ignores_blank_line(self):
        self.check(input="   \n vhdl file text", expected="vhdl file text")

    def test_ignore_line_comment(self):
        self.check("-- my comment\n my vhdl text", expected="my vhdl text")

    def test_ignore_trailing_comments(self):
        self.check("some vhdl code -- comment", expected="some vhdl code")

    def test_ignore_dashed_line(self):
        self.check("------\nsome text", expected="some text")

    def test_ignore_trailing_dashed_line(self):
        self.check("some text ------", expected="some text")

    def test_ignore_trailing_empty_comment(self):
        self.check("some text --", expected="some text")
