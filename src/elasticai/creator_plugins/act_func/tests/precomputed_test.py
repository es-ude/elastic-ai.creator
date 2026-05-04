import random
from pathlib import Path

import cocotb
from cocotb.triggers import Timer

import elasticai.creator_plugins.act_func as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["precomputed.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="ACT_PRECOMPUTED",
    cocotb_test_module="elasticai.creator_plugins.act_func.tests.precomputed_tb",
    params={"BITWIDTH": 4, "NUM_VALUES": 4},
)


@cocotb.test()
async def act_precomputed_random(dut):
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned() - 1)
    for _ in range(9):
        A = random.randint(-valrange, valrange - 1)
        dut.A.value = A
        await Timer(2, unit="step")
        # assert dut.Q.value.to_signed() == dut.selector.value


@cocotb.test()
async def act_precomputed_sweep(dut):
    data_in = [
        -(2 ** (dut.BITWIDTH.value.to_unsigned() - 1)) + val
        for val in range(2 ** (dut.BITWIDTH.value.to_unsigned()))
    ]
    data_out = []
    for val in data_in:
        dut.A.value = val
        await Timer(2, unit="step")
        data_out.append(dut.Q.value.to_signed())
        # assert dut.Q.value.to_signed() == dut.selector.value

    print("data_in:", data_in)
    print("data_out:", data_out)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
