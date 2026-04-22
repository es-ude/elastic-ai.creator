import random
from pathlib import Path

import cocotb
import elasticai.creator_plugins.mult as test_dut
from cocotb.triggers import Timer

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["adder_half.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="ADDER_LUT_HALF",
    cocotb_test_module="elasticai.creator_plugins.mult.tests.adder_half_tb",
)


def adder_model(a: int, b: int) -> tuple[int, int]:
    return a ^ b, a & b


@cocotb.test()
async def adder_half_basic(dut):
    A = 1
    B = 1
    dut.A.value = A
    dut.B.value = B
    await Timer(2, units="step")
    assert (dut.Q.value, dut.Cout.value) == (0, 1)


@cocotb.test()
async def adder_half_random(dut):
    for _ in range(10):
        A = random.randint(0, 1)
        B = random.randint(0, 1)
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit="step")
        assert (dut.Q.value, dut.Cout.value) == adder_model(A, B)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
