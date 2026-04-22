import random
import cocotb
from cocotb.triggers import Timer
from pathlib import Path

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 
import elasticai.creator_plugins.mult as test_dut


cocotb_settings = dict(
    src_files=["mult_dadda_u6.v", 'adder_full.v', 'adder_half.v'],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='MULT_DADDA_UNSIGNED_6BIT',
    cocotb_test_module="elasticai.creator_plugins.mult.tests.mult_dadda_u6_tb",
    params={}
)


@cocotb.test()
async def mult_lut_first(dut):
    dut.A.value = 1
    dut.B.value = 1
    await Timer(2, unit='step')
    output = dut.Q.value
    assert output == 1


@cocotb.test()
async def mult_lut_second(dut):
    dut.A.value = 2
    dut.B.value = 4
    await Timer(2, unit='step')
    output = dut.Q.value
    assert output == 8


@cocotb.test()
async def mult_lut_random(dut):
    num_bit = 6
    valrange = 2**(num_bit)-1
    for _ in range(2**(int(num_bit/2))):
        A = random.randint(0, valrange)
        B = random.randint(0, valrange)
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit='step')
        output = dut.Q.value
        assert output == A * B


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
