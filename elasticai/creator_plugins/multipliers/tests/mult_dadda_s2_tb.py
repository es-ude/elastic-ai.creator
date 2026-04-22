import random
from pathlib import Path

import cocotb
import elasticai.creator_plugins.mult as test_dut
from cocotb.triggers import Timer

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["mult_dadda_s2.v", "adder_full.v", "adder_half.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="MULT_DADDA_SIGNED_2BIT",
    cocotb_test_module="elasticai.creator_plugins.mult.tests.mult_dadda_s2_tb",
    params={},
)


@cocotb.test()
async def mult_lut_first(dut):
    dut.A.value = 1
    dut.B.value = 1
    await Timer(2, unit="step")
    output = dut.Q.value
    assert output.to_signed() == 1


@cocotb.test()
async def mult_lut_second(dut):
    dut.A.value = -2
    dut.B.value = 1
    await Timer(2, unit="step")
    output = dut.Q.value
    assert output.to_signed() == -2


@cocotb.test()
async def mult_lut_random(dut):
    num_bit = 2
    valrange = 2 ** (num_bit - 1)
    for _ in range(2 ** (int(num_bit / 2))):
        A = random.randint(-valrange, valrange - 1)
        B = random.randint(-valrange, valrange - 1)
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit="step")
        output = dut.Q.value
        assert output.to_signed() == A * B


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
