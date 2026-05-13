from random import randint

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.types import LogicArray

import elasticai.creator_plugins.adders as adders
import elasticai.creator_plugins.multipliers as multipliers
from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin


def build_testdata(
    bitwidth: int, num_params: int, is_signed: bool, repeats: int = 16
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    arith = int_arithmetic(
        total_bits=bitwidth - int(np.log2(num_params)) + 1, signed=is_signed
    )

    data = [
        [
            randint(arith.minimum_as_integer, arith.maximum_as_integer)
            for _ in range(num_params)
        ]
        for _ in range(repeats)
    ]
    weights = [
        [
            randint(arith.minimum_as_integer, arith.maximum_as_integer)
            for _ in range(num_params)
        ]
        for _ in range(repeats)
    ]
    bias = [
        randint(arith.minimum_as_integer, arith.maximum_as_integer)
        for _ in range(repeats)
    ]
    return data, weights, bias


def build_testdata_max(
    bitwidth: int, num_params: int, is_signed: bool, repeats: int = 16
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)

    data = [
        [arith.maximum_as_integer for _ in range(num_params)] for _ in range(repeats)
    ]
    weights = [
        [arith.maximum_as_integer for _ in range(num_params)] for _ in range(repeats)
    ]
    bias = [arith.maximum_as_integer for _ in range(repeats)]
    return data, weights, bias


def build_testdata_min(
    bitwidth: int, num_params: int, is_signed: bool, repeats: int = 16
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)

    data = [
        [arith.minimum_as_integer for _ in range(num_params)] for _ in range(repeats)
    ]
    weights = [
        [arith.maximum_as_integer for _ in range(num_params)] for _ in range(repeats)
    ]
    bias = [arith.minimum_as_integer for _ in range(repeats)]
    return data, weights, bias


def model_mac(
    bias: int, weights: list, data: list, bitwidth: int, is_signed: bool
) -> int:
    arith = int_arithmetic(total_bits=2 * bitwidth, signed=is_signed)
    return arith.clamp(bias + int(np.sum(np.array(weights) * np.array(data))))


@cocotb.test()
@eai_testbench
async def mac_calculation(
    dut,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
    data_in: list[list[int]],
    weights_in: list[list[int]],
    bias_in: list[int],
):
    period_clk = 2

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

    conv = int_converter(total_bits=bitwidth, signed=is_signed)

    for data0, gain0, bias0 in zip(data_in, weights_in, bias_in):
        await RisingEdge(dut.CLK_SYS)
        val_data = ""
        val_gain = ""
        for data, gain in zip(data0, gain0):
            val_data += conv.integer_to_binary_string_verilog(data).split("b")[-1]
            val_gain += conv.integer_to_binary_string_verilog(gain).split("b")[-1]

        dut.IN_BIAS.value = bias0
        dut.IN_WEIGHTS.value = LogicArray(val_gain)
        dut.IN_DATA.value = LogicArray(val_data)

        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0
        await RisingEdge(dut.CLK_SYS)

        await RisingEdge(dut.DATA_RDY)
        assert dut.DATA_RDY.value == 1
        await RisingEdge(dut.CLK_SYS)
        check = model_mac(
            bias=bias0,
            weights=gain0,
            data=data0,
            bitwidth=bitwidth,
            is_signed=is_signed,
        )
        result = dut.OUT_DATA.value.to_signed()

        if check != result:
            print("\n")
            print(bias0)
            print(data0)
            print(gain0)
            print(check, dut.mac_out.value.to_signed(), dut.OUT_DATA.value.to_signed())
        assert dut.OUT_DATA.value.to_signed() == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("num_params", [4, 8, 16])
@pytest.mark.parametrize("num_mult", [1, 2, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_dsp(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    data0, weights0, bias0 = build_testdata(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
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
def test_mac_lut(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    data0, weights0, bias0 = build_testdata(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_lut_signed.v")
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
@pytest.mark.parametrize("num_params", [4])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_dadda(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    data0, weights0, bias0 = build_testdata(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac.v")
    cocotb_test_fixture.add_srcs_from_package(
        multipliers, f"verilog/mult_dadda_s{bitwidth}.v"
    )
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
@pytest.mark.parametrize("num_params", [4, 8])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_lut_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mac",
        id="",
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        packages=["mac"],
        path2save=build_dir,
    )
    load_and_plugin(
        type="mult_lut_signed",
        id="",
        params={"BITWIDTH": bitwidth},
        packages=["multipliers", "adders"],
        path2save=build_dir,
    )

    data0, weights0, bias0 = build_testdata(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )
    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4])
@pytest.mark.parametrize("num_params", [4, 8])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_dsp_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mac",
        id="",
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        packages=["mac"],
        path2save=build_dir,
    )
    load_and_plugin(
        type="mult_dsp_signed",
        id="",
        params={"BITWIDTH": bitwidth},
        packages=["multipliers", "adders"],
        path2save=build_dir,
    )

    data0, weights0, bias0 = build_testdata(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )
    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("num_params", [16, 32])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_clamp_overflow(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    data0, weights0, bias0 = build_testdata_max(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
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
@pytest.mark.parametrize("num_params", [16, 32, 64])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_clamp_underflow(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    data0, weights0, bias0 = build_testdata_min(
        bitwidth=bitwidth, num_params=num_params, is_signed=is_signed, repeats=32
    )
    cocotb_test_fixture.write(
        {"data_in": data0, "weights_in": weights0, "bias_in": bias0}
    )

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        defines={},
    )
