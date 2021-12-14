import unittest
from os.path import exists
from elasticai.generator.generate_sigmoid import main


class GenerateSigmoidTestBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        main()
        self.generated_testbench_file = open('../testbench/generated_sigmoid_tb.vhd', 'r')
        self.lines = self.generated_testbench_file.readlines()

    def tearDown(self) -> None:
        self.generated_testbench_file.close()

    def test_generate_file(self) -> None:
        self.assertTrue(exists('../testbench/generated_sigmoid_tb.vhd'))
