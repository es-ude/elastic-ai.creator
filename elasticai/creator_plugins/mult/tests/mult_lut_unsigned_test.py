from unittest import TestCase, main
from copy import deepcopy
from shutil import rmtree

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    check_cocotb_test_result,
    run_cocotb_sim_for_src_dir
)
from elasticai.creator_plugins.mult.tests.mult_lut_unsigned_tb import cocotb_settings


class VerilogMultiplierLut(TestCase):
    @classmethod
    def setUpClass(cls):
        rmtree(find_project_root() / 'build_sim', ignore_errors=True)

    def test_2bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 2}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_4bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 4}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 6}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_8bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_10bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 10}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_12bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 12}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_14bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 14}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit(self):
        set0 = deepcopy(cocotb_settings)
        set0['params'] = {'BITWIDTH': 16}
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    main()
