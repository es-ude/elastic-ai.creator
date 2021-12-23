import unittest
from os.path import exists
from elasticai.generator.functions.generate_sigmoid_vhd import main


class GenerateSigmoidVhdTest(unittest.TestCase):
    def setUp(self) -> None:
        main()
        self.generated_file = open("../../source/sigmoid.vhd", "r")
        self.generated_lines = self.generated_file.readlines()
        self.expected_file = open("vhdFiles/sigmoid_for_testing.vhd", "r")
        self.expected_lines = self.expected_file.readlines()

    def tearDown(self) -> None:
        self.generated_file.close()
        self.expected_file.close()

    def test_generate_file(self) -> None:
        self.assertTrue(exists("../../source/sigmoid.vhd"))

    def test_compare_files(self) -> None:
        # clean each file from empty lines and lines which are just comment
        self.expected_lines = [
            line.strip()
            for line in self.expected_lines
            if not line.startswith("--") and not line.isspace()
        ]

        self.generated_lines = [
            line.strip()
            for line in self.generated_lines
            if not line.startswith("--") and not line.isspace()
        ]

        self.assertEqual(self.generated_lines, self.expected_lines)
        self.assertEqual(len(self.generated_lines), len(self.expected_lines))
