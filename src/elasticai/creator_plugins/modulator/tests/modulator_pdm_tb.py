import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge
from pathlib import Path

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.modulator as test_dut


cocotb_settings = dict(
    src_files=['modulator_pdm.v'],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='PULSE_DENSITY_MODULATOR',
    cocotb_test_module='elasticai.creator_plugins.modulator.tests.modulator_pdm_tb',
    params={"MOD_ORDER":4, "CNTWIDTH_CLK":8}
)


@cocotb.test()
async def modulator_pdm_tb(dut):
    period_clk = 5

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.REF_VAL.value = 3
    dut.REF_CLK.value = 2

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit='ns').start())
    await Timer(4 * period_clk, unit='ns')
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    await Timer(4 * period_clk, unit='ns')
    dut.EN.value = 1

    cnt_positive = 0
    for _ in range(100):
        await RisingEdge(dut.CLK_STREAM)
        cnt_positive += 1 if dut.PDM_STREAM.value else 0

    print(cnt_positive)
    assert cnt_positive > 0


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
