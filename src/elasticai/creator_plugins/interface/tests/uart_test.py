import unittest
from shutil import rmtree
from copy import deepcopy

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    run_cocotb_sim_for_src_dir,
    check_cocotb_test_result
)
from elasticai.creator_plugins.interface.tests.uart_com_tb import cocotb_settings as uart_mod_sets
from elasticai.creator_plugins.interface.tests.uart_com_mirror_tb import cocotb_settings as uart_mir_sets
from elasticai.creator_plugins.interface.tests.uart_com_fifo_tb import cocotb_settings as uart_fifo_sets
from elasticai.creator_plugins.interface.tests.uart_com_fifo_mid_tb import cocotb_settings as uart_mid_sets


class VerilogUart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rmtree(find_project_root() / 'build_sim', ignore_errors=True)

    def test_baud115200_ovr4_8bit_direct(self):
        sets0 = deepcopy(uart_mod_sets)
        sets0['params'] = {'BITRATE': 217, 'NSAMP': 4, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud115200_ovr8_8bit_direct(self):
        sets0 = deepcopy(uart_mod_sets)
        sets0['params'] = {'BITRATE': 109, 'NSAMP': 8, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud115200_ovr4_5bit_direct(self):
        sets0 = deepcopy(uart_mod_sets)
        sets0['params'] = {'BITRATE': 217, 'NSAMP': 4, 'BITWIDTH': 5}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud115200_ovr8_5bit_direct(self):
        sets0 = deepcopy(uart_mod_sets)
        sets0['params'] = {'BITRATE': 109, 'NSAMP': 8, 'BITWIDTH': 5}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr4_8bit_direct(self):
        sets0 = deepcopy(uart_mod_sets)
        sets0['params'] = {'BITRATE': 27, 'NSAMP': 4, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_8bit_direct(self):
        sets0 = deepcopy(uart_mod_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud115200_ovr4_8bit_mirror(self):
        sets0 = deepcopy(uart_mir_sets)
        sets0['params'] = {'BITRATE': 217, 'NSAMP': 4, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud115200_ovr8_8bit_mirror(self):
        sets0 = deepcopy(uart_mir_sets)
        sets0['params'] = {'BITRATE': 109, 'NSAMP': 8, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr4_8bit_mirror(self):
        sets0 = deepcopy(uart_mir_sets)
        sets0['params'] = {'BITRATE': 27, 'NSAMP': 4, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_8bit_mirror(self):
        sets0 = deepcopy(uart_mir_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_8bit_fifo2(self):
        sets0 = deepcopy(uart_fifo_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'FIFO_SIZE': 2, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_5bit_fifo2(self):
        sets0 = deepcopy(uart_fifo_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'FIFO_SIZE': 2, 'BITWIDTH': 5}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_8bit_fifo4(self):
        sets0 = deepcopy(uart_fifo_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'FIFO_SIZE': 4, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_8bit_fifo8(self):
        sets0 = deepcopy(uart_fifo_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'FIFO_SIZE': 8, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir(**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_baud921600_ovr8_8bit_fifo3_middleware(self):
        sets0 = deepcopy(uart_mid_sets)
        sets0['params'] = {'BITRATE': 14, 'NSAMP': 8, 'FIFO_SIZE': 3, 'BITWIDTH': 8}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    unittest.main()
