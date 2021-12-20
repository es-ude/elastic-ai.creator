import unittest
from os.path import exists
from elasticai.generator.generate_tanh_testbench import main


class GenerateTanhTestBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        main(path_to_testbench='../../testbench/')
        self.generated_testbench_file = open('../../testbench/tanh_tb.vhd', 'r')
        self.real_testbench_file = open('tanh_for_testing_tb.vhd', 'r')

    def tearDown(self) -> None:
        self.generated_testbench_file.close()
        self.real_testbench_file.close()

    def test_generate_file(self) -> None:
        self.assertTrue(exists('../../testbench/tanh_tb.vhd'))

    def test_equality_of_real_file_and_generated_file(self) -> None:
        for line1, line2 in zip(self.generated_testbench_file.readlines(), self.real_testbench_file.readlines()):
            self.assertEqual(line1, line2)