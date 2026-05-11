import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.multipliers.utils import (
    generate_mult_testdata,
    load_and_plugin,
)


@cocotb.test()
@eai_testbench
async def mult_calculation(dut, bitwidth: int, is_signed: bool):
    period_clk = 2
    dut.CLK.value = 0
    dut.RSTN.value = 1
    dut.START_FLAG.value = 0
    dut.A.value = 0
    dut.B.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK, period_clk, unit="ns").start())
    await Timer(4 * period_clk, unit="ns")
    for idx in range(4):
        await RisingEdge(dut.CLK)
        dut.RSTN.value = idx % 2

    await RisingEdge(dut.CLK)
    dut.RSTN.value = 1
    await Timer(4 * period_clk, unit="ns")

    # Apply tests
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    input_a = generate_mult_testdata(arith)
    input_b = generate_mult_testdata(arith)

    arith_mult = int_arithmetic(total_bits=2 * bitwidth, signed=is_signed)
    for a, b in zip(input_a, input_b):
        await Timer(period_clk, unit="ns")
        dut.A.value = arith.to_twos(a)
        dut.B.value = arith.to_twos(b)

        dut.START_FLAG.value = 1
        await Timer(period_clk, unit="ns")
        dut.START_FLAG.value = 0
        await Timer(period_clk, unit="ns")

        # await FallingEdge(dut.DRDY)
        await Timer((bitwidth + 1) * period_clk, unit="ns")

        check = arith_mult.clamp(a * b)
        print(a, b, dut.Q.value, check)
        if is_signed:
            assert dut.Q.value.to_signed() == check
        else:
            assert dut.Q.value.to_unsigned() == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 4, 6, 8, 10, 12])
@pytest.mark.parametrize("is_signed", [True])
def test_mult_booth_signed(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    cocotb_test_fixture.set_top_module_name("MULT_SIGNED")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 4, 6, 8, 10, 12])
@pytest.mark.parametrize("is_signed", [False])
def test_mult_booth_unsigned(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    cocotb_test_fixture.set_top_module_name("MULT_UNSIGNED")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("is_signed", [False, True])
def test_mult_booth_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    module_name = "mult_booth_signed" if is_signed else "mult_booth_unsigned"
    top_name = "MULT_SIGNED" if is_signed else "MULT_UNSIGNED"

    load_and_plugin(
        type=module_name,
        id=f"{bitwidth}",
        params={"BITWIDTH": bitwidth},
        packages=["multipliers"],
        path2save=build_dir,
    )

    cocotb_test_fixture.set_top_module_name(top_name)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
