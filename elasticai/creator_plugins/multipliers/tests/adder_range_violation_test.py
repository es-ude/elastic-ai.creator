from copy import deepcopy
from shutil import rmtree
from unittest import TestCase, main

from elasticai.creator_plugins.multipliers.tests.adder_range_violation_dtct_tb import (
    cocotb_settings,
)

from elasticai.creator.file_generation import find_project_root
from elasticai.creator.testing.cocotb_runner import (
    check_cocotb_test_result,
    run_cocotb_sim_for_src_dir,
)


class VerilogAdderViolationDetection(TestCase):
    @classmethod
    def setUpClass(cls):
        rmtree(find_project_root() / "build_sim", ignore_errors=True)

    def test_unsigned_4bit_add0(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 4, "NUM_ADDERS": 0, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_4bit_add1(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 4, "NUM_ADDERS": 1, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_4bit_add2(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 4, "NUM_ADDERS": 2, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_8bit_add0(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 8, "NUM_ADDERS": 0, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_8bit_add2(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 8, "NUM_ADDERS": 2, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_8bit_add3(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 8, "NUM_ADDERS": 3, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_12bit_add0(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 12, "NUM_ADDERS": 0, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_unsigned_12bit_add4(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 12, "NUM_ADDERS": 4, "SIGNED": 0}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_4bit_add0(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 4, "NUM_ADDERS": 0, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_4bit_add1(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 4, "NUM_ADDERS": 1, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_4bit_add2(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 4, "NUM_ADDERS": 2, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_8bit_add0(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 8, "NUM_ADDERS": 0, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_8bit_add2(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 8, "NUM_ADDERS": 2, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_8bit_add3(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 8, "NUM_ADDERS": 3, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_12bit_add0(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 12, "NUM_ADDERS": 0, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())

    def test_signed_12bit_add4(self):
        sets = deepcopy(cocotb_settings)
        sets["params"] = {"BITWIDTH": 12, "NUM_ADDERS": 4, "SIGNED": 1}
        run_cocotb_sim_for_src_dir(**sets)
        self.assertTrue(check_cocotb_test_result())


if __name__ == "__main__":
    main()
