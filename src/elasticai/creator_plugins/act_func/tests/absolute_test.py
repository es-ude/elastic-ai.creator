from pathlib import Path
from random import randint

import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.testing import CocotbTestFixture, eai_testbench


def act_absolute(a: int) -> int:
    return abs(a)


@cocotb.test()
@eai_testbench
async def xrange_positive(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    valrange = 2 ** (bitwidth - 1)
    A = randint(0, valrange - 1)
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_absolute(A)


@cocotb.test()
@eai_testbench
async def xrange_negative(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    valrange = 2 ** (bitwidth - 1)
    A = randint(-valrange + 1, 0)
    dut.A.value = A
    await Timer(2, unit="step")
    assert dut.Q.value.to_signed() == act_absolute(A)


@cocotb.test()
@eai_testbench
async def xrange_random(dut):
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    valrange = 2 ** (bitwidth - 1)
    for _ in range(16):
        A = randint(-valrange + 1, valrange - 1)
        dut.A.value = A
        await Timer(2, unit="step")
        assert dut.Q.value.to_signed() == act_absolute(A)


@pytest.mark.simulation
def test_absolute(cocotb_test_fixture: CocotbTestFixture):
    # cocotb_test_fixture.add_srcs_from_package("act_func", "*.v")
    import elasticai.creator_plugins.act_func as dut0

    artifact_dir = Path(dut0.__file__).parent

    build_dir = artifact_dir / "verilog"

    cocotb_test_fixture.set_top_module_name("ACT_ABSOLUTE")
    cocotb_test_fixture.add_srcs(build_dir / "absolute.v")
    cocotb_test_fixture.run(params={}, defines={})
