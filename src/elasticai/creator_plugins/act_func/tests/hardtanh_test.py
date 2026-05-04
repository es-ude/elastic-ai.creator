import random
from pathlib import Path

import cocotb
from cocotb.triggers import Timer
from fxpmath import Fxp

import elasticai.creator_plugins.act_func as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir

cocotb_settings = dict(
    src_files=["hardtanh.v"],
    path2src=Path(test_dut.__file__).parent / "verilog",
    top_module_name="ACT_HARDTANH",
    cocotb_test_module="elasticai.creator_plugins.act_func.tests.hardtanh_tb",
    params={"BITWIDTH": 4},
)


def act_hardtanhsign(a: int, bitwidth: int, fraction: int) -> int:
    range = Fxp(val=[-1.0, +1.0], n_word=bitwidth, n_frac=fraction, signed=True)
    xmin = int(range[0].val)
    xmax = int(range[1].val)
    return xmin if a < xmin else (xmax if a > xmax else a)


@cocotb.test()
async def act_access_positive(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    A = random.randint(0, 2 ** (bitwidth - 1) - 1)
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_hardtanhsign(A, bitwidth, 2)


@cocotb.test()
async def act_access_negative(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    A = random.randint(-(2 ** (bitwidth - 1)), 0)
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_hardtanhsign(A, bitwidth, 2)


@cocotb.test()
async def act_random(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()

    valrange = 2 ** (bitwidth - 1)
    for _ in range(9):
        A = random.randint(-valrange, valrange - 1)
        dut.A.value = A
        await Timer(2, unit="step")
        assert dut.Q.value.to_signed() == act_hardtanhsign(A, bitwidth, 2)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir(**cocotb_settings)
