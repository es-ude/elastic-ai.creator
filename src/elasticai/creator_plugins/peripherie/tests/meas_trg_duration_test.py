import math

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.peripherie.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def measure_trigger_duration_tb(dut, repeat: int, cycles: int):
    period_clk = 5

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.START.value = 0
    dut.SIG.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await Timer(4 * period_clk, unit="ns")
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    await Timer(4 * period_clk, unit="ns")

    for _ in range(repeat):
        for _ in range(cycles):
            await RisingEdge(dut.CLK_SYS)
        dut.SIG.value = 1
        for _ in range(cycles):
            await RisingEdge(dut.CLK_SYS)
        dut.SIG.value = 0

    for _ in range(repeat):
        await RisingEdge(dut.CLK_SYS)

    dut.SIG.value = 0

    reps = dut.CNT_RPT.value.to_unsigned()
    assert reps == repeat
    width = dut.CNT_PERIOD.value.to_unsigned()
    assert width == reps * cycles


@pytest.mark.simulation
@pytest.mark.parametrize("repeat, cycles", [(4, 32), (8, 8)])
def test_meas_trg_duration(
    cocotb_test_fixture: CocotbTestFixture, repeat: int, cycles: int
):
    cocotb_test_fixture.set_top_module_name("MEAS_TRG_DURATION")
    cocotb_test_fixture.write({"repeat": repeat, "cycles": cycles})
    cocotb_test_fixture.run(
        params={
            "CNTWIDTH_TRG": math.log2(repeat * cycles),
            "CNTWIDTH_OVR": math.log2(repeat) + 1,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("repeat, cycles", [(8, 8)])
def test_meas_trg_duration_build(
    cocotb_test_fixture: CocotbTestFixture, repeat: int, cycles: int
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="measure_trigger",
        id="0",
        params={
            "CNTWIDTH_TRG": math.log2(repeat * cycles),
            "CNTWIDTH_OVR": math.log2(repeat) + 1,
        },
        packages=["peripherie"],
        path2save=build_dir,
    )

    cocotb_test_fixture.set_top_module_name("MEASURE_TRIGGER_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
