import unittest
from os.path import exists
from elasticai.generator.generate_sigmoid_vhd import main


class GenerateSigmoidTestBenchTest(unittest.TestCase):
    def setUp(self) -> None:
        main()
        self.generated_file = open('../source/generated_sigmoid.vhd', 'r')
        self.generated_lines = self.generated_file.readlines()

    def tearDown(self) -> None:
        self.generated_file.close()

    def test_generate_file(self) -> None:
        self.assertTrue(exists('../source/generated_sigmoid.vhd'))




