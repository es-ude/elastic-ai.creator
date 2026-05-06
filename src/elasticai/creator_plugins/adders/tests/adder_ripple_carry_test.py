import random

import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.adders.utils import load_and_plugin


def adder_model(a: int, b: int, bitwidth: int, signed: bool) -> int:
    arith = int_arithmetic(total_bits=bitwidth + 1, signed=signed)
    return arith.clamp(a + b)


@cocotb.test()
@eai_testbench
async def adder_truthtable(dut, bitwidth: int, is_signed: bool):
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    limits = [arith.minimum_as_integer, arith.maximum_as_integer]
    for _ in range(256):
        A = random.randint(a=limits[0], b=limits[1])
        B = random.randint(a=limits[0], b=limits[1])
        dut.A.value = arith.to_twos(A)
        dut.B.value = arith.to_twos(B)
        await Timer(2, unit="step")
        test_val = adder_model(A, B, bitwidth, is_signed)
        if is_signed:
            assert dut.Q.value.to_signed() == test_val
        else:
            assert dut.Q.value.to_unsigned() == test_val


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [2, 5, 8, 12, 15, 18])
@pytest.mark.parametrize("is_signed", [False, True])
def test_adder_ripple_carry(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    top_name = (
        "ADDER_RIPPLE_CARRY_SIGNED" if is_signed else "ADDER_RIPPLE_CARRY_UNSIGNED"
    )
    cocotb_test_fixture.set_top_module_name(top_name)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("adders", "verilog/adder_*.v")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8, 10])
@pytest.mark.parametrize("is_signed", [False, True])
def test_adder_ripple_carry_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    module_name = (
        "adder_ripple_carry_signed" if is_signed else "adder_ripple_carry_unsigned"
    )
    top_name = f"{module_name.upper()}_{bitwidth}"

    load_and_plugin(
        type=module_name,
        id=f"{bitwidth}",
        params={"BITWIDTH": bitwidth},
        packages=["adders"],
        path2save=build_dir,
    )
    cocotb_test_fixture.set_top_module_name(top_name)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
