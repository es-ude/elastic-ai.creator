from pathlib import Path

import pytest

from elasticai.creator.file_generation.resource_utils import get_full_path
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

_path2src = Path(
    get_full_path("tests.unit_tests.testing", "cocotb_runner_tb.py")
).parent
_path2tb = "tests.unit_tests.testing.cocotb_runner_test"


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_without_defines_without_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        defines={},
        params={},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_without_defines_with_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_with_defines_without_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={"DEFINE_TEST": True},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_with_defines_with_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={"DEFINE_TEST": True},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_without_defines_without_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_without_defines_with_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_with_defines_without_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={"DEFINE_TEST": True},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_with_defines_with_params() -> None:
    sets = dict(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2, "OFFSET": 2},
        defines={"DEFINE_TEST": True},
        en_debug_mode=True,
    )
    run_cocotb_sim_for_src_dir(**sets)
