from copy import deepcopy
from shutil import rmtree
from unittest import TestCase, main

from elasticai.creator_plugins.multipliers.tests.adder_full_tb import cocotb_settings

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    check_cocotb_test_result,
    run_cocotb_sim_for_src_dir,
)


class VerilogAdderFull(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rmtree(find_project_root() / "build_sim", ignore_errors=True)

    def test_initial(self):
        sets = deepcopy(cocotb_settings)
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())


if __name__ == "__main__":
    main()
