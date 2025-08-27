from os.path import exists
from pathlib import Path

import pytest

from elasticai.creator.file_generation.resource_utils import (
    get_full_path,
)
from elasticai.creator.testing.cocotb_runner import (
    run_cocotb_sim,
    run_cocotb_sim_for_src_dir,
)

_path2src = str(
    Path(get_full_path("tests.unit_tests.testing", "cocotb_runner_tb.py")).parent
)
_path2tb = "tests.unit_tests.testing.cocotb_runner_tb"


@pytest.mark.simulation
def test_verilog_without_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        defines={},
        params={},
    )


@pytest.mark.simulation
def test_verilog_without_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={},
    )


@pytest.mark.simulation
def test_verilog_with_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={"DEFINE_TEST": True},
    )


@pytest.mark.simulation
def test_verilog_with_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={"DEFINE_TEST": True},
    )


@pytest.mark.simulation
def test_verilog_with_waveforms_default() -> None:
    path = run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
    )
    assert exists(path / "waveforms.vcd")


@pytest.mark.simulation
def test_verilog_with_waveforms_external() -> None:
    path = run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.v"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
        waveform_save_dst="build_test",
    )
    assert exists(path / "waveforms.vcd")


@pytest.mark.simulation
def test_vhdl_without_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
    )


@pytest.mark.simulation
def test_vhdl_without_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={},
    )


@pytest.mark.simulation
def test_vhdl_with_defines_without_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={"DEFINE_TEST": True},
    )


@pytest.mark.simulation
def test_vhdl_with_defines_with_params() -> None:
    run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={"BITWIDTH": 4, "SCALE": 2},
        defines={"DEFINE_TEST": True},
    )


@pytest.mark.simulation
def test_verilog_with_run_direct() -> None:
    run_cocotb_sim(
        src_files=[f"{_path2src}/cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        params={},
        defines={},
    )


@pytest.mark.simulation
def test_vhdl_with_run_direct() -> None:
    run_cocotb_sim(
        src_files=[f"{_path2src}/cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        params={},
        defines={},
    )


@pytest.mark.simulation
def test_vhdl_with_waveforms_normal() -> None:
    path = run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
        waveform_save_dst="",
    )
    assert exists(path / "waveforms.vcd")


@pytest.mark.simulation
def test_vhdl_with_waveforms_external() -> None:
    path = run_cocotb_sim_for_src_dir(
        src_files=["cocotb_runner_tb.vhd"],
        top_module_name="COCOTB_TEST",
        cocotb_test_module=_path2tb,
        path2src=_path2src,
        params={},
        defines={},
        waveform_save_dst="build_test",
    )
    assert exists(path / "waveforms.vcd")
