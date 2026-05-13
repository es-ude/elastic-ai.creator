from random import randint

import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac_fxp import load_and_plugin


@cocotb.test()
@eai_testbench
async def int_transform_random_positive(dut, bitwidth: int):
    arith = int_arithmetic(total_bits=bitwidth, signed=True)
    for _ in range(2 ** (bitwidth - 2)):
        val = randint(0, arith.maximum_as_integer)
        dut.A.value = val
        await Timer(2, unit="step")
        assert dut.Q.value.to_signed() == val


@cocotb.test()
@eai_testbench
async def int_transform_random_negative(dut, bitwidth: int):
    arith = int_arithmetic(total_bits=bitwidth, signed=True)
    for _ in range(2 ** (bitwidth - 2)):
        val = randint(arith.minimum_as_integer, -1)
        dut.A.value = arith.to_twos(val)
        await Timer(2, unit="step")
        assert dut.Q.value.to_signed() == val


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 6, 8, 9, 12])
def test_uint_to_int(cocotb_test_fixture: CocotbTestFixture, bitwidth: int):
    cocotb_test_fixture.set_top_module_name("UNSIGNED_TO_SIGNED")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [6, 10])
def test_uint_to_int_build(cocotb_test_fixture: CocotbTestFixture, bitwidth: int):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="uint_to_int",
        id=f"{bitwidth}",
        params={"BITWIDTH": bitwidth},
        packages=["mac_fxp"],
        path2save=build_dir,
    )

    cocotb_test_fixture.set_top_module_name(f"UINT_TO_INT_{bitwidth}")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
