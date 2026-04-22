import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge
from pathlib import Path

from elasticai.creator.testing.cocotb_runner import (
    run_cocotb_sim_for_src_dir
)
import elasticai.creator_plugins.mult as test_dut


cocotb_settings = dict(
    src_files=["mult_booth_signed.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='MULT_BOOTH_SIGNED',
    cocotb_test_module="elasticai.creator_plugins.mult.tests.mult_booth_signed_tb",
    params={'BITWIDTH': 4},
)


@cocotb.test()
async def mult_booth_signed_simple(dut):
    period_clk = 2
    dut.CLK.value = 0
    dut.nRST.value = 1
    dut.START_FLAG.value = 0
    dut.DATA_A.value = 0
    dut.DATA_B.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK, period_clk, unit='ns').start())
    await Timer(4*period_clk, unit='ns')
    for idx in range(4):
        await RisingEdge(dut.CLK)
        dut.nRST.value = idx % 2

    await RisingEdge(dut.CLK)
    dut.nRST.value = 1
    await Timer(4 * period_clk, unit='ns')

    # Start Testing
    dut.START_FLAG.value = 1
    dut.DATA_A.value = -1
    dut.DATA_B.value = -1
    await Timer(2*period_clk, unit='ns')
    dut.START_FLAG.value = 0

    # Checking
    await RisingEdge(dut.DRDY)
    await Timer(2*period_clk, unit='ns')
    output = dut.DOUT.value
    assert output.signed_integer == 1


@cocotb.test()
async def mult_booth_signed_random(dut):
    period_clk = 2
    dut.CLK.value = 0
    dut.nRST.value = 1
    dut.START_FLAG.value = 0
    dut.DATA_A.value = 0
    dut.DATA_B.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK, period_clk, unit='ns').start())
    await Timer(4*period_clk, unit='ns')
    for idx in range(4):
        await RisingEdge(dut.CLK)
        dut.nRST.value = idx % 2

    await RisingEdge(dut.CLK)
    dut.nRST.value = 1
    await Timer(4 * period_clk, unit='ns')

    # Apply tests
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned() - 1)
    for _ in range(256):
        A = random.randint(0, valrange - 1)
        B = random.randint(-valrange, valrange - 1)
        await Timer(2 * period_clk, unit='ns')
        dut.DATA_A.value = A
        dut.DATA_B.value = B

        dut.START_FLAG.value = 1
        await Timer(2 * period_clk, unit='ns')
        dut.START_FLAG.value = 0

        await RisingEdge(dut.DRDY)
        await Timer(2 * period_clk, unit='ns')
        assert dut.DOUT.value.signed_integer == A * B


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
