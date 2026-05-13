from random import randint

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.types import LogicArray

from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
import elasticai.creator_plugins.multipliers as multipliers
import elasticai.creator_plugins.adders as adders


def model_mac(bias: int, weights: list, data: list) -> int:
    return bias + int(
            np.sum(np.array(weights) * np.array(data))
        )


@cocotb.test()
@eai_testbench
async def mac_calculation(
    dut, bitwidth: int, num_params: int, num_mult: int, is_signed: bool
):
    period_clk = 2
    period_data = 4 * int(num_params / num_mult) + 2
    repeat = 100

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 1
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.IN_BIAS.value = 0
    dut.IN_WEIGHTS.value = 0
    dut.IN_DATA.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await Timer(4 * period_clk, unit="ns")
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
    await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    for _ in range(4):
        await RisingEdge(dut.CLK_SYS)

    # Apply data and test
    dut.EN.value = 1
    for _ in range(4):
        await RisingEdge(dut.CLK_SYS)
    cocotb.start_soon(Clock(dut.DO_CALC, period_data, unit="ns").start())

    conv = int_converter(total_bits=bitwidth, signed=is_signed)
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    for _ in range(repeat):
        await RisingEdge(dut.CLK_SYS)
        val_bias = randint(arith.minimum_as_integer, arith.maximum_as_integer)
        val_data0 = [
            randint(arith.minimum_as_integer, arith.maximum_as_integer)
            for _ in range(num_params)
        ]
        val_gain0 = [
            randint(arith.minimum_as_integer, arith.maximum_as_integer)
            for _ in range(num_params)
        ]

        val_data = ""
        val_gain = ""
        for data, gain in zip(val_data0, val_gain0):
            val_data += conv.integer_to_binary_string_verilog(data).split("b")[-1]
            val_gain += conv.integer_to_binary_string_verilog(gain).split("b")[-1]

        dut.IN_BIAS.value = val_bias
        dut.IN_WEIGHTS.value = LogicArray(val_gain)
        dut.IN_DATA.value = LogicArray(val_data)

        await RisingEdge(dut.DATA_VALID)
        assert dut.DATA_VALID.value == 1
        await RisingEdge(dut.CLK_SYS)
        check = model_mac(val_bias, val_gain0, val_data0)
        print(val_bias)
        print(val_data0)
        print(val_gain0)
        print(check, dut.mac_out.value.to_signed(), dut.OUT_DATA.value.to_signed())
        print("\n")
        assert dut.OUT_DATA.value.to_signed() == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("num_params", [4, 8, 16])
@pytest.mark.parametrize("num_mult", [1, 2, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_fxp_dsp(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    cocotb_test_fixture.set_top_module_name("MAC_FXP")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac_fxp", "verilog/mac_fxp.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, f"verilog/mult_dadda_s{bitwidth}.v")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("num_params", [2, 4, 8, 16])
@pytest.mark.parametrize("num_mult", [1, 2, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_fxp_lut(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    cocotb_test_fixture.set_top_module_name("MAC_FXP")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac_fxp", "verilog/mac_fxp.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, f"verilog/mult_dadda_s{bitwidth}.v")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("num_params", [2, 4, 6])
@pytest.mark.parametrize("num_mult", [1, 2, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_fxp_dadda(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    cocotb_test_fixture.set_top_module_name("MAC_FXP")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac_fxp", "verilog/mac_fxp.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, f"verilog/mult_dadda_s{bitwidth}.v")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        defines={},
    )
