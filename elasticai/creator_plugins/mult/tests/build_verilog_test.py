import unittest
from shutil import rmtree
from os import makedirs
from os.path import exists

from denspp.translate import get_path_to_project_start
from elasticai.creator_plugins.ir_generator_verilog import build_verilog_design


class BuildVerilogAdder(unittest.TestCase):
    path2build = f"{get_path_to_project_start()}/build_temp"

    @classmethod
    def setUpClass(cls):
        rmtree(cls.path2build, ignore_errors=True)
        makedirs(cls.path2build, exist_ok=True)

    def test_build_mult_lut_signed_general_wo_tb(self):
        build_verilog_design(
            type="mult_lut_signed",
            id='0',
            params={'BITWIDTH': 3},
            packages=['mult'],
            build_tb=False,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_signed_0.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_lut_signed_general_w_tb(self):
        build_verilog_design(
            type="mult_lut_signed",
            id='1',
            params={'BITWIDTH': 3},
            packages=['mult'],
            build_tb=True,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_signed_1.v"))
        self.assertTrue(exists(f"{self.path2build}/mult_lut_signed_1_tb.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_dadda_signed_general_wo_tb(self):
        build_verilog_design(
            type="mult_lut_signed",
            id='2',
            params={'BITWIDTH': 4},
            packages=['mult'],
            build_tb=False,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_signed_2.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_dadda_signed_general_w_tb(self):
        build_verilog_design(
            type="mult_lut_signed",
            id='3',
            params={'BITWIDTH': 4},
            packages=['mult'],
            build_tb=True,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_signed_3.v"))
        self.assertTrue(exists(f"{self.path2build}/mult_lut_signed_3_tb.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_lut_unsigned_general_wo_tb(self):
        build_verilog_design(
            type="mult_lut_unsigned",
            id='0',
            params={'BITWIDTH': 4},
            packages=['mult'],
            build_tb=False,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_unsigned_0.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_lut_unsigned_general_w_tb(self):
        build_verilog_design(
            type="mult_lut_unsigned",
            id='1',
            params={'BITWIDTH': 4},
            packages=['mult'],
            build_tb=True,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_unsigned_1.v"))
        self.assertTrue(exists(f"{self.path2build}/mult_lut_unsigned_1_tb.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_dadda_unsigned_general_wo_tb(self):
        build_verilog_design(
            type="mult_lut_unsigned",
            id='2',
            params={'BITWIDTH': 6},
            packages=['mult'],
            build_tb=False,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_unsigned_2.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))

    def test_build_mult_dadda_unsigned_general_w_tb(self):
        build_verilog_design(
            type="mult_lut_unsigned",
            id='3',
            params={'BITWIDTH': 6},
            packages=['mult'],
            build_tb=True,
            path2save=self.path2build
        )
        self.assertTrue(exists(f"{self.path2build}/mult_lut_unsigned_3.v"))
        self.assertTrue(exists(f"{self.path2build}/mult_lut_unsigned_3_tb.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_full.v"))
        self.assertTrue(exists(f"{self.path2build}/adder_half.v"))


if __name__ == '__main__':
    unittest.main()
