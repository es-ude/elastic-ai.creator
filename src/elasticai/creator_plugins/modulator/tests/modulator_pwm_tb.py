import cocotb
import random
from cocotb.triggers import Timer, RisingEdge
from cocotb.clock import Clock
from pathlib import Path

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.modulator as test_dut


cocotb_settings = dict(
    src_files=['modulator_pwm.v'],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='PULSE_WIDTH_MODULATOR',
    cocotb_test_module='elasticai.creator_plugins.modulator.tests.modulator_pwm_tb',
    params={"PERIOD_NUM_CYCLE": 16}
)


@cocotb.test()
async def modulator_pwm_tb(dut):
    period_clk = 5
    duty_cycle_test = random.randint(a=1, b=dut.PERIOD_NUM_CYCLE.value.to_unsigned()-1)

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DUTY_CYCLE.value = duty_cycle_test

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
    for _ in range(3):
        for _ in range(dut.PERIOD_NUM_CYCLE.value.to_unsigned()):
            await RisingEdge(dut.CLK_SYS)
            cnt_positive += 1 if dut.PWM_STREAM.value else 0
    assert cnt_positive / 2 == duty_cycle_test


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
