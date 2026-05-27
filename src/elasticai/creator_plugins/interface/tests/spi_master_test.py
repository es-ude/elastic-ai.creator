import unittest
from shutil import rmtree
from copy import deepcopy

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    run_cocotb_sim_for_src_dir,
    check_cocotb_test_result
)

from elasticai.creator_plugins.interface.tests.spi_master_tb import cocotb_settings


class VerilogSpiMaster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rmtree(find_project_root() / 'build_sim', ignore_errors=True)

    def test_6bit_cpol0_cpha0_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 0, 'CPHA': 0, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol0_cpha1_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 0, 'CPHA': 1, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol1_cpha0_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 1, 'CPHA': 0, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol1_cpha1_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 1, 'CPHA': 1, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol0_cpha0_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 0, 'CPHA': 0, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol0_cpha1_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 0, 'CPHA': 1, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol1_cpha0_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 1, 'CPHA': 0, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_6bit_cpol1_cpha1_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 6, 'CPOL': 1, 'CPHA': 1, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol0_cpha0_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 0, 'CPHA': 0, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol0_cpha1_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 0, 'CPHA': 1, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol1_cpha0_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 1, 'CPHA': 0, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol1_cpha1_lsb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 1, 'CPHA': 1, 'MSB': 0, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol0_cpha0_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 0, 'CPHA': 0, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol0_cpha1_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 0, 'CPHA': 1, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol1_cpha0_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 1, 'CPHA': 0, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())

    def test_16bit_cpol1_cpha1_msb(self):
        sets0 = deepcopy(cocotb_settings)
        sets0['params'] = {'BITWIDTH': 16, 'CPOL': 1, 'CPHA': 1, 'MSB': 1, 'SPI_DIV_CLK': 2}
        run_cocotb_sim_for_src_dir (**sets0)
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    unittest.main()
