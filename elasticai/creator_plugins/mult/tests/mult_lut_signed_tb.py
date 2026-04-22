import random
import cocotb
from cocotb.triggers import Timer
from pathlib import Path

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 
import elasticai.creator_plugins.mult as test_dut


cocotb_settings = dict(
    src_files=["mult_lut_signed.v", 'adder_full.v', 'adder_half.v'],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='MULT_LUT_SIGNED',
    cocotb_test_module="elasticai.creator_plugins.mult.tests.mult_lut_signed_tb",
    params={'BITWIDTH': 8},
    en_debug_mode=True
)


@cocotb.test()
async def mult_lut_access(dut):
    dut.A.value = 1
    dut.B.value = -2
    await Timer(2, unit='step')
    output = dut.Q.value
    assert output.to_signed() == -2


@cocotb.test()
async def mult_lut_random(dut):
    valrange = 2**(dut.BITWIDTH.value.to_unsigned() -1)
    for _ in range(256):
        A = random.randint(-valrange, valrange-1)
        B = random.randint(-valrange, valrange-1)
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit='step')
        output = dut.Q.value
        assert output.to_signed() == A * B


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
