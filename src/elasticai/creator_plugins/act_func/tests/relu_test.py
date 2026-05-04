import random
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

import elasticai.creator_plugins.act_func as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["relu.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="ACT_RELU",
    cocotb_test_module="elasticai.creator_plugins.act_func.tests.relu_tb",
    params={"BITWIDTH": 4},
)


def act_relu(a: int) -> int:
    return 0 if a < 0 else a


@cocotb.test()
async def act_access_positive(dut):
    A = 1
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_relu(A)


@cocotb.test()
async def act_access_negative(dut):
    A = -1
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_relu(A)


@cocotb.test()
async def act_random(dut):
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned() - 1)
    for _ in range(9):
        A = random.randint(-valrange, valrange - 1)
        dut.A.value = A
        await Timer(2, unit="step")
        assert dut.Q.value.to_signed() == act_relu(A)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
