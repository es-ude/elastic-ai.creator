from random import randint

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

import elasticai.creator_plugins.multipliers as multipliers
from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin


def build_testdata(
    bitwidth_data: int,
    bitwidth_wght: int,
    num_params: int,
    is_signed: bool,
    repeats: int = 16,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    arith0 = int_arithmetic(total_bits=bitwidth_data, signed=is_signed)
    arith1 = int_arithmetic(total_bits=bitwidth_wght, signed=is_signed)

    data = [
        [
            randint(arith0.minimum_as_integer, arith0.maximum_as_integer)
            for _ in range(num_params)
        ]
        for _ in range(repeats)
    ]
    data.append([arith0.maximum_as_integer for _ in range(num_params)])
    data.append([arith0.minimum_as_integer for _ in range(num_params)])
    weights0 = [
        randint(arith1.minimum_as_integer, arith1.maximum_as_integer)
        for _ in range(repeats)
    ]
    weightsd = [
        [
            randint(arith1.minimum_as_integer, arith1.maximum_as_integer)
            for _ in range(num_params)
        ]
        for _ in range(repeats)
    ]
    weightsd.append([arith1.maximum_as_integer for _ in range(num_params)])
    weightsd.append([arith1.maximum_as_integer for _ in range(num_params)])
    bias = [
        randint(arith0.minimum_as_integer, arith0.maximum_as_integer)
        for _ in range(repeats)
    ]
    bias.append(arith0.maximum_as_integer)
    bias.append(arith0.minimum_as_integer)
    return data, weightsd, weights0, bias


def get_full_weights(weightsd: list[list[int]], weights0: list[int]) -> list[list[int]]:
    weights_full = list()

    for wdelta, winit in zip(weightsd, weights0):
        weights_reconstructed = list()
        for idx, val in enumerate(wdelta):
            weights_reconstructed.append(winit + val)
        weights_full.append(weights_reconstructed)
    return weights_full


def model_mac(
    bias: int, weights: list, data: list, bitwidth: int, is_signed: bool
) -> int:
    arith = int_arithmetic(total_bits=2 * bitwidth, signed=is_signed)
    return arith.clamp(bias + int(np.sum(np.array(weights) * np.array(data))))


@cocotb.test()
@eai_testbench
async def mac_delta_tb(
    dut,
    bitwidth: int,
    deltawidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
    data_in: list[list[int]],
    weights_in: list[list[int]],
    weights_delta: list[list[int]],
    weights_init: list[int],
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
    dut.INITIAL_WEIGHT.value = 0

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

    conv_data = int_converter(total_bits=bitwidth, signed=is_signed)
    conv_wght = int_converter(total_bits=deltawidth, signed=is_signed)

    for data0, gain0, bias0, gain1, init1 in zip(
        data_in, weights_in, bias_in, weights_delta, weights_init
    ):
        await RisingEdge(dut.CLK_SYS)
        val_data = ""
        val_gain = ""
        for data, gain in zip(data0, gain1):
            val_data += conv_data.integer_to_binary_string_verilog(data).split("b")[-1]
            val_gain += conv_wght.integer_to_binary_string_verilog(gain).split("b")[-1]

        dut.IN_BIAS.value = bias0
        dut.INITIAL_WEIGHT.value = init1
        dut.IN_WEIGHTS.value = LogicArray(val_gain)
        dut.IN_DATA.value = LogicArray(val_data)

        await RisingEdge(dut.CLK_SYS)
        t0 = get_sim_time("ns")
        dut.DO_CALC.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0
        await RisingEdge(dut.CLK_SYS)

        await RisingEdge(dut.DATA_RDY)
        t1 = get_sim_time("ns")
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

        dt = int((t1 - t0) / period_clk)
        assert dt == int(num_params / num_mult) + 2
        if check != result:
            print("\n")
            print(gain0)
            print("\n")
            print(init1)
            print(gain1)
            print(check, dut.mac_out.value.to_signed(), dut.OUT_DATA.value.to_signed())
        assert dut.OUT_DATA.value.to_signed() == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, deltawidth", [(8, 4), (12, 4)])
@pytest.mark.parametrize("num_params", [8, 32])
@pytest.mark.parametrize("num_mult", [1, 2, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_delta_reference_dsp(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    deltawidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    data0, weightsd, weights0, bias0 = build_testdata(
        bitwidth_data=bitwidth,
        bitwidth_wght=deltawidth,
        num_params=num_params,
        is_signed=is_signed,
        repeats=32,
    )
    weights_full = get_full_weights(weightsd=weightsd, weights0=weights0)

    cocotb_test_fixture.write(
        {
            "data_in": data0,
            "weights_in": weights_full,
            "bias_in": bias0,
            "weights_delta": weightsd,
            "weights_init": weights0,
        }
    )

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_delta_reference.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    # cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_DELTAWIDTH": deltawidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, deltawidth", [(12, 4)])
@pytest.mark.parametrize("num_params", [32, 128])
@pytest.mark.parametrize("num_mult", [1, 2, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_delta_reference_dsp_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    deltawidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mac_delta_reference",
        id="",
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_DELTAWIDTH": deltawidth,
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
        packages=["multipliers"],
        path2save=build_dir,
    )

    data0, weightsd, weights0, bias0 = build_testdata(
        bitwidth_data=bitwidth,
        bitwidth_wght=deltawidth,
        num_params=num_params,
        is_signed=is_signed,
        repeats=32,
    )
    weights_full = get_full_weights(weightsd=weightsd, weights0=weights0)

    cocotb_test_fixture.write(
        {
            "data_in": data0,
            "weights_in": weights_full,
            "bias_in": bias0,
            "weights_delta": weightsd,
            "weights_init": weights0,
        }
    )
    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
