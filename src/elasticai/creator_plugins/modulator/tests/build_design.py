import unittest
from os import makedirs
from os.path import exists
from shutil import rmtree

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir, check_cocotb_test_result
from elasticai.creator_plugins.ir_generator_verilog import build_verilog_design


class BuildVerilogModulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path2build = find_project_root() / 'build_temp'
        rmtree(cls.path2build, ignore_errors=True)
        makedirs(cls.path2build, exist_ok=True)

    def test_build_modulator_pdm_wo_tb(self):
        build_verilog_design(
            type='modulator_pdm',
            id='0',
            params={"MOD_ORDER": 4, "CNTWIDTH_CLK": 8},
            packages=['modulator'],
            build_tb=False,
            path2save=self.path2build
        )
        self.assertTrue(exists(self.path2build / 'modulator_pdm_0.v'))

    def test_build_modulator_pdm_w_tb(self):
        build_verilog_design(
            type='modulator_pdm',
            id='1',
            params={"MOD_ORDER":4, "CNTWIDTH_CLK":8},
            packages=['modulator'],
            build_tb=True,
            path2save=self.path2build
        )
        self.assertTrue(exists(self.path2build / 'modulator_pdm_1.v'))
        self.assertTrue(exists(self.path2build / 'modulator_pdm_1_tb.v'))

    def test_run_modulator_pdm(self):
        run_cocotb_sim_for_src_dir(
            src_files=['modulator_pdm_0.v'],
            path2src=self.path2build,
            top_module_name='MODULATOR_PDM_0',
            cocotb_test_module="elasticai.creator_plugins.modulator.tests.modulator_pdm_tb"
        )
        self.assertTrue(check_cocotb_test_result())

    def test_build_modulator_pwm_wo_tb(self):
        build_verilog_design(
            type='modulator_pwm',
            id='0',
            params={"PERIOD_NUM_CYCLE": 16},
            packages=['modulator'],
            build_tb=False,
            path2save=self.path2build
        )
        self.assertTrue(exists(self.path2build / 'modulator_pwm_0.v'))

    def test_build_modulator_pwm_w_tb(self):
        build_verilog_design(
            type='modulator_pwm',
            id='1',
            params={"PERIOD_NUM_CYCLE": 16},
            packages=['modulator'],
            build_tb=True,
            path2save=self.path2build
        )
        self.assertTrue(exists(self.path2build / 'modulator_pwm_1.v'))
        self.assertTrue(exists(self.path2build / 'modulator_pwm_1_tb.v'))

    def test_run_modulator_pwm(self):
        run_cocotb_sim_for_src_dir(
            src_files=['modulator_pwm_0.v'],
            path2src=self.path2build,
            top_module_name='MODULATOR_PWM_0',
            cocotb_test_module="elasticai.creator_plugins.modulator.tests.modulator_pwm_tb"
        )
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    unittest.main()
