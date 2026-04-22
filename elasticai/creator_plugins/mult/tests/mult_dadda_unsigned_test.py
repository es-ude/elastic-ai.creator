from unittest import TestCase, main
from copy import deepcopy
from shutil import rmtree

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    check_cocotb_test_result,
    run_cocotb_sim_for_src_dir
)


class VerilogMultiplierDaddaUnsigned(TestCase):
    @classmethod
    def setUpClass(cls):
        rmtree(find_project_root() / 'build_sim', ignore_errors=True)

    def test_6bit(self):
        from elasticai.creator_plugins.mult.tests.mult_dadda_u6_tb import cocotb_settings
        set0 = deepcopy(cocotb_settings)
        run_cocotb_sim_for_src_dir(**set0)
        self.assertTrue(check_cocotb_test_result())


if __name__ == '__main__':
    main()
