import random
import cocotb
from cocotb.triggers import Timer
from pathlib import Path

import elasticai.creator_plugins.mult as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=["adder_full.v", "adder_half.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='ADDER_LUT_FULL',
    cocotb_test_module="elasticai.creator_plugins.mult.tests.adder_full_tb"
)


def adder_model(a: int, b: int, cin: int) -> tuple[int, int]:
    return a ^ b ^ cin, 1 if a+b+cin > 1 else 0


@cocotb.test()
async def adder_full_basic(dut):
    A = 1
    B = 0
    Cin = 0
    dut.A.value = A
    dut.B.value = B
    dut.Cin.value = Cin
    await Timer(2, unit='step')
    assert (dut.Q.value, dut.Cout.value) == adder_model(A, B, Cin)


@cocotb.test()
async def adder_full_random(dut):
    for _ in range(9):
        A = random.randint(0, 1)
        B = random.randint(0, 1)
        Cin = random.randint(0, 1)
        dut.A.value = A
        dut.B.value = B
        dut.Cin.value = Cin
        await Timer(2, unit='step')
        assert (dut.Q.value, dut.Cout.value) == adder_model(A, B, Cin)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
