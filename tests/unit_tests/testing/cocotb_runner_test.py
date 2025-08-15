from pathlib import Path

import pytest

from os.path import join, exists

from elasticai.creator.file_generation.resource_utils import get_full_path, find_project_root
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir, run_cocotb_sim


_path2src = str(Path(
    get_full_path("tests.unit_tests.testing", "cocotb_runner_tb.py")
).parent)
_path2tb = "tests.unit_tests.testing.cocotb_runner_tb"


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_without_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        defines={},
        params={}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_without_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_with_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={"DEFINE_TEST": True}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_with_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={"DEFINE_TEST": True}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_with_waveforms() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
        build_waveforms=True
    )
    assert exists(join(find_project_root(), "build_sim", "COCOTB_TEST.fst"))


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_without_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_without_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_with_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={"DEFINE_TEST": True}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_with_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={"DEFINE_TEST": True}
    )

@pytest.mark.simulation
@pytest.mark.slow
def test_verilog_with_run_direct() -> None:
    run_cocotb_sim(
        src_files=[f"{_path2src}/cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        params={},
        defines={}
    )


@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_with_run_direct() -> None:
    run_cocotb_sim(
        src_files=[f"{_path2src}/cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        params={},
        defines={}
    )

@pytest.mark.simulation
@pytest.mark.slow
def test_vhdl_with_waveforms() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
        build_waveforms=True
    )
    assert exists(join(find_project_root(), "build_sim", "cocotb_test.vcd"))
