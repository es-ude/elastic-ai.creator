import random
from math import floor
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

import elasticai.creator_plugins.act_func as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["prelu.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="ACT_PRELU",
    cocotb_test_module="elasticai.creator_plugins.act_func.tests.prelu_tb",
    params={"BITWIDTH": 4, "SCALING": 2},
)


def act_prelu(a: int, scale: int) -> int:
    return floor(a * 2**-scale) if a < 0 else a


@cocotb.test()
async def act_access_positive(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    scale = dut.SCALING.value.to_unsigned()

    A = random.randint(0, 2 ** (bitwidth - 1) - 1)
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_prelu(A, scale)


@cocotb.test()
async def act_access_negative(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    scale = dut.SCALING.value.to_unsigned()

    A = random.randint(-(2 ** (bitwidth - 1)), 0)
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_prelu(A, scale)


@cocotb.test()
async def act_random(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    scale = dut.SCALING.value.to_unsigned()
    valrange = 2 ** (bitwidth - 1)

    for _ in range(9):
        A = random.randint(-valrange, valrange - 1)
        dut.A.value = A
        await Timer(2, unit="step")
        assert dut.Q.value.to_signed() == act_prelu(A, scale)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
