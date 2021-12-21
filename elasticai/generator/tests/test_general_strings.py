import unittest
from elasticai.generator.general_strings import get_libraries_string, get_architecture_header_string, \
    get_architecture_end_string


class GeneralStringsTest(unittest.TestCase):
    def test_generate_libraries(self) -> None:
        expected_import_lines = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;               -- for type conversions"
        ]
        lib_string = get_libraries_string()
        for i in range(0, 3):
            self.assertEqual(expected_import_lines[i], lib_string.splitlines()[i])

    def test_generate_libraries_with_math_lib(self) -> None:
        expected_import_lines = [
            "library ieee;",
            "use ieee.std_logic_1164.all;",
            "use ieee.numeric_std.all;               -- for type conversions",
            "use ieee.math_real.all;"
        ]
        lib_string = get_libraries_string(math_lib=True)
        for i in range(0, 4):
            self.assertEqual(expected_import_lines[i], lib_string.splitlines()[i])

    def test_generate_architecture_header(self) -> None:
        expected_architecture_header_line = "architecture behav of sigmoid_tb is"
        architecture_header_string = get_architecture_header_string(architecture_name="behav", component_name="sigmoid_tb")
        self.assertEqual(expected_architecture_header_line, architecture_header_string.splitlines()[0])

    def test_generate_architecture_end(self) -> None:
        expected_test_architecture_end_line = "end behav ; -- behav"
        architecture_end_string = get_architecture_end_string(architecture_name="behav")
        self.assertEqual(expected_test_architecture_end_line, architecture_end_string.splitlines()[0])
