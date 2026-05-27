import unittest
from shutil import rmtree
from os import makedirs
from os.path import exists, join
from elasticai.creator.file_generation import find_project_root
from elasticai.creator_plugins.ir_generator_verilog import build_verilog_design


class BuildVerilogInterface(unittest.TestCase):
    path2build = find_project_root() / "build_temp"

    params_spi_slave = {
        'BITWIDTH': 8,
        'CPHA': 0,
        'CPOL': 0,
        'MSB': 1
    }
    params_spi_master = {
        'BITWIDTH': 8,
        'CPHA': 0,
        'CPOL': 0,
        'MSB': 1,
        'SPI_DIV_CLK': 4
    }
    params_uart = {
        'BITRATE': 27,
        'NSAMP': 4,
        "FIFO_SIZE": 3,
        "BITWIDTH": 8,
        'BITWIDTH_CMDS': 2,
        'BITWIDTH_ADR': 6,
        'BITWIDTH_DATA': 16,
    }

    @classmethod
    def setUpClass(cls):
        rmtree(cls.path2build, ignore_errors=True)
        makedirs(cls.path2build, exist_ok=True)

    def test_build_spi_slave_woclk_wotb(self):
        type_name = "spi_slave_without_clk"
        build_verilog_design(
            type=type_name,
            id='0',
            params=self.params_spi_slave,
            build_tb=False,
            packages=["interface"],
            path2save=self.path2build
        )
        chck_files = [f"{type_name}_0.v", "spi_middleware.v"]
        chck = [exists(join(self.path2build, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_spi_slave_with_clk_wotb(self):
        type_name = "spi_slave_with_clk"
        build_verilog_design(
            type=type_name,
            id='0',
            params=self.params_spi_slave,
            build_tb=False,
            packages=["interface"],
            path2save=self.path2build
        )
        chck_files = [f"{type_name}_0.v", "spi_middleware.v"]
        chck = [exists(join(self.path2build, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_spi_master_wotb(self):
        type_name = "spi_master"
        build_verilog_design(
            type=type_name,
            id='0',
            params=self.params_spi_master,
            build_tb=False,
            packages=["interface"],
            path2save=self.path2build
        )
        chck_files = [f"{type_name}_0.v"]
        chck = [exists(join(self.path2build, file)) for file in chck_files]
        self.assertTrue(all(chck))

    def test_build_uart(self):
        type_name = "uart"
        build_verilog_design(
            type=type_name,
            id='0',
            params=self.params_uart,
            build_tb=True,
            packages=["interface"],
            path2save=self.path2build
        )
        chck_files = [f"{type_name}_0.v", f"{type_name}_fifo.v", f"{type_name}_middleware.v"]
        chck = [exists(join(self.path2build, file)) for file in chck_files]
        self.assertTrue(all(chck))


if __name__ == '__main__':
    unittest.main()
