from unittest import TestCase, main
from copy import deepcopy
from shutil import rmtree

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    check_cocotb_test_result,
    run_cocotb_sim_for_src_dir
)
from elasticai.creator_plugins.mult.tests.adder_half_tb import cocotb_settings


class VerilogAdderHalf(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rmtree(find_project_root() / 'build_sim', ignore_errors=True)

    def test_initial(self):
        sets = deepcopy(cocotb_settings)
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    main()
