import random
from pathlib import Path

import cocotb
import elasticai.creator_plugins.mult as test_dut
from cocotb.triggers import Timer

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["mult_lut_unsigned.v", "adder_full.v", "adder_half.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="MULT_LUT_UNSIGNED",
    cocotb_test_module="elasticai.creator_plugins.mult.tests.mult_lut_unsigned_tb",
    params={"BITWIDTH": 8},
)


@cocotb.test()
async def mult_lut_access(dut):
    dut.A.value = 1
    dut.B.value = 1
    await Timer(2, unit="step")
    output = dut.Q.value
    assert output == 1


@cocotb.test()
async def mult_lut_random(dut):
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned()) - 1
    for _ in range(256):
        A = random.randint(0, valrange)
        B = random.randint(0, valrange)
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit="step")
        output = dut.Q.value
        assert output == A * B


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
