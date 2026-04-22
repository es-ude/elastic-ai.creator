import random
import cocotb
import numpy as np
from pathlib import Path
from fxpmath import Fxp
from cocotb.clock import Clock
from cocotb.types import LogicArray
from cocotb.triggers import Timer, RisingEdge

import elasticai.creator_plugins.mac as test_mac
import elasticai.creator_plugins.mult as test_mult
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim


_path2mac = Path(test_mac.__file__).parent / 'verilog'
_path2mult = Path(test_mult.__file__).parent / 'verilog'


cocotb_settings_dsp = dict(
    src_files=[
        _path2mac / "mac_fxp_dsp.v",
    ],
    top_module_name='MAC_DSP',
    cocotb_test_module="elasticai.creator_plugins.mac.tests.mac_xxx_tb",
    params={'INPUT_BITWIDTH': 6, 'INPUT_NUM_DATA': 3, 'NUM_MULT_PARALLEL': 1}
)
cocotb_settings_lut = dict(
    src_files=[
        _path2mac / "mac_fxp_lut.v",
        _path2mult / "mult_lut_signed.v",
        _path2mult / "adder_full.v",
        _path2mult / "adder_half.v",
    ],
    top_module_name='MAC_LUT',
    cocotb_test_module="elasticai.creator_plugins.mac.tests.mac_xxx_tb",
    params={'INPUT_BITWIDTH': 6, 'INPUT_NUM_DATA': 3, 'NUM_MULT_PARALLEL': 1}
)


@cocotb.test()
async def mac_access(dut):
    period_clk = 2
    period_data = 100
    repeat = 100
    numwidth = dut.INPUT_NUM_DATA.value.to_unsigned()
    bitwidth = dut.INPUT_BITWIDTH.value.to_unsigned()
    valrange = 2 ** (bitwidth - numwidth - 1)

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.IN_BIAS.value = 0
    dut.IN_WEIGHTS.value = 0
    dut.IN_DATA.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit='ns').start())
    await Timer(4 * period_clk, unit='ns')
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    for _ in range(4):
        await RisingEdge(dut.CLK_SYS)

    # Apply data and test
    dut.EN.value = 1
    for _ in range(4):
        await RisingEdge(dut.CLK_SYS)
    cocotb.start_soon(Clock(dut.DO_CALC, period_data, unit='ns').start())
    await RisingEdge(dut.CLK_SYS)
    for _ in range(repeat):
        val_bias = random.randint(-valrange, valrange - 1)
        val_data0 = [random.randint(-valrange, valrange - 1) for _ in range(numwidth)]
        val_gain0 = [random.randint(-valrange, valrange - 1) for _ in range(numwidth)]
        val_data1 = Fxp(val=val_data0, signed=True, n_word=bitwidth, n_frac=0).bin()
        val_gain1 = Fxp(val=val_gain0, signed=True, n_word=bitwidth, n_frac=0).bin()
        val_data = ''
        val_gain = ''
        for data, gain in zip(val_data1, val_gain1):
            val_data += data
            val_gain += gain

        dut.IN_BIAS.value = val_bias
        dut.IN_WEIGHTS.value = LogicArray(val_gain)
        dut.IN_DATA.value = LogicArray(val_data)

        await RisingEdge(dut.DATA_VALID)
        assert dut.DATA_VALID.value == 1
        await RisingEdge(dut.CLK_SYS)
        assert dut.OUT_DATA.value.to_signed() == val_bias + int(np.sum(np.array(val_gain0) * np.array(val_data0)))


if __name__ == "__main__":
    run_cocotb_sim(**cocotb_settings_dsp)
    run_cocotb_sim(**cocotb_settings_lut)
