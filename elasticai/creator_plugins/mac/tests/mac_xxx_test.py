from unittest import TestCase, main
from copy import deepcopy
from shutil import rmtree

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    check_cocotb_test_result,
    run_cocotb_sim
)
from elasticai.creator_plugins.mac.tests.mac_xxx_tb import cocotb_settings_dsp, cocotb_settings_lut


class VerilogSignedMAC(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rmtree(find_project_root() / 'build_temp', ignore_errors=True)
        rmtree(find_project_root() / 'build_sim', ignore_errors=True)

    def test_dsp_4bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 4, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_6bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 6, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_8bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 8, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_8bit_2in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 8, 'INPUT_NUM_DATA': 2, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_8bit_3in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 8, 'INPUT_NUM_DATA': 3, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_10bit_1in_1mult(self):
        set0 =  deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 10, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_10bit_4in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 10, 'INPUT_NUM_DATA': 4, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_10bit_4in_2mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 10, 'INPUT_NUM_DATA': 4, 'NUM_MULT_PARALLEL': 2}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_12bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 12, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_14bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 14, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_16bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 16, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_dsp_16bit_8in_4mult(self):
        set0 = deepcopy(cocotb_settings_dsp)
        set0['params'] = {'INPUT_BITWIDTH': 16, 'INPUT_NUM_DATA': 8, 'NUM_MULT_PARALLEL': 4}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_4bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 4, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_6bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 6, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_8bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 8, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_8bit_2in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 8, 'INPUT_NUM_DATA': 2, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_10bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 10, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_10bit_2in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 10, 'INPUT_NUM_DATA': 2, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_10bit_4in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 10, 'INPUT_NUM_DATA': 4, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_12bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 12, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_12bit_4in_2mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 12, 'INPUT_NUM_DATA': 4, 'NUM_MULT_PARALLEL': 2}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_14bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 14, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_16bit_1in_1mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 16, 'INPUT_NUM_DATA': 1, 'NUM_MULT_PARALLEL': 1}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_lut_16bit_8in_4mult(self):
        set0 = deepcopy(cocotb_settings_lut)
        set0['params'] = {'INPUT_BITWIDTH': 16, 'INPUT_NUM_DATA': 8, 'NUM_MULT_PARALLEL': 4}
        run_cocotb_sim(**set0)
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    main()
