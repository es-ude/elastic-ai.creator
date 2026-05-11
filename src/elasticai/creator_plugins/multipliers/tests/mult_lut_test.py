import cocotb
import pytest
from cocotb.triggers import Timer

import elasticai.creator_plugins.adders as adders
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.multipliers.utils import (
    generate_mult_testdata,
    load_and_plugin,
)


@cocotb.test()
@eai_testbench
async def mult_truthtable(dut, bitwidth: int, is_signed: bool):
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    input_a = generate_mult_testdata(arith)
    input_a.extend([arith.minimum_as_integer, arith.maximum_as_integer])
    input_b = generate_mult_testdata(arith)
    input_b.extend([arith.maximum_as_integer, arith.minimum_as_integer])

    arith_mult = int_arithmetic(total_bits=2 * bitwidth, signed=is_signed)
    for a, b in zip(input_a, input_b):
        dut.A.value = a
        dut.B.value = b
        await Timer(2, unit="step")
        if is_signed:
            assert dut.Q.value.to_signed() == arith_mult.clamp(a * b)
        else:
            assert dut.Q.value.to_unsigned() == arith_mult.clamp(a * b)


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [True])
def test_mult_lut_signed(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    cocotb_test_fixture.set_top_module_name("MULT_SIGNED")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8, 12, 16])
@pytest.mark.parametrize("is_signed", [False])
def test_mult_lut_unsigned(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    cocotb_test_fixture.set_top_module_name("MULT_UNSIGNED")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [3, 16])
@pytest.mark.parametrize("is_signed", [False, True])
def test_mult_lut_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, is_signed: bool
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    module_name = "mult_lut_signed" if is_signed else "mult_lut_unsigned"
    top_name = "MULT_SIGNED" if is_signed else "MULT_UNSIGNED"

    load_and_plugin(
        type=module_name,
        id=f"{bitwidth}",
        params={"BITWIDTH": bitwidth},
        packages=["multipliers", "adders"],
        path2save=build_dir,
    )

    cocotb_test_fixture.set_top_module_name(top_name)
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
