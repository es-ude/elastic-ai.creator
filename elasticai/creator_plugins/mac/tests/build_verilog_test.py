import unittest
from shutil import rmtree
from os import makedirs
from os.path import exists, join
from denspp.translate import get_path_to_project_start
from elasticai.creator_plugins.ir_generator_verilog import build_verilog_design


class BuildVerilogMath(unittest.TestCase):
    path2build = f"{get_path_to_project_start()}/build_temp"
    params_mac = {
        'BITWIDTH': 4,
        'INPUT_BITWIDTH': 4,
        'INPUT_NUM_DATA': 2,
        'NUM_MULT_PARALLEL': 1
    }

    @classmethod
    def setUpClass(cls):
        rmtree(cls.path2build, ignore_errors=True)
        makedirs(cls.path2build, exist_ok=True)

    def test_build_mac_fpga(self):
        build_verilog_design(
            type="mac_fxp_fpga",
            id='0',
            params=self.params_mac,
            build_tb=False,
            packages=["mac"],
            path2save=self.path2build
        )
        chck_files = ["mac_fxp_fpga_0.v"]
        chck = [exists(join(self.path2build, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_mac_asic(self):
        build_verilog_design(
            type="mac_fxp_asic",
            id='0',
            params=self.params_mac,
            build_tb=False,
            packages=["mac", 'mult'],
            path2save=self.path2build
        )
        chck_files = ["mac_fxp_asic_0.v"]
        chck = [exists(join(self.path2build, file)) for file in chck_files]
        self.assertTrue(all(chck))


if __name__ == '__main__':
    unittest.main()
