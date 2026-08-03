from itertools import batched
from random import randint

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

import elasticai.creator_plugins.mac_delta as mac_delta
import elasticai.creator_plugins.multipliers as multipliers
from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac.tests.mac_core_test import model_mac


def build_testdata(
    bitwidth_data: int,
    bitwidth_wght: int,
    num_params: int,
    is_signed: bool,
    repeats: int = 16,
) -> tuple[list[list[int]], list[list[int]], list[int], list[int]]:
    arith0 = int_arithmetic(total_bits=bitwidth_data, signed=is_signed)
    arith1 = int_arithmetic(total_bits=bitwidth_wght - 2, signed=is_signed)

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
        for _ in range(repeats)
    ]
    weightsd = [
        [
            randint(arith1.minimum_as_integer, arith1.maximum_as_integer)
            if idx > 0
            else 0
            for idx in range(num_params)
        ]
        for _ in range(repeats)
    ]
    weightsd.append(
        [arith1.maximum_as_integer if idx > 0 else 0 for idx in range(num_params)]
    )
    weightsd.append(
        [arith1.maximum_as_integer if idx > 0 else 0 for idx in range(num_params)]
    )
    bias = [
        randint(arith0.minimum_as_integer, arith0.maximum_as_integer)
        for _ in range(repeats)
    ]
    bias.append(arith0.maximum_as_integer)
    bias.append(arith0.minimum_as_integer)
    return data, weightsd, weights0, bias


def get_full_weights_consecutive(
    weightsd: list[list[int]], weights0: list[int], bitwidth: int
) -> list[list[int]]:
    arith = int_arithmetic(total_bits=bitwidth, signed=True)
    weights_full = list()
    for wdelta, winit in zip(weightsd, weights0):
        weights_reconstructed = [winit]
        for idx, val in enumerate(wdelta[1:]):
            val = weights_reconstructed[-1] + val
            if val < arith.minimum_as_integer:
                val = arith.maximum_as_integer + (val % arith.minimum_as_integer) + 1
            elif val > arith.maximum_as_integer:
                val = arith.minimum_as_integer + (val % arith.maximum_as_integer) - 1
            weights_reconstructed.append(val)
        weights_full.append(weights_reconstructed)
    return weights_full


def get_full_weights_reference(
    weightsd: list[list[int]], weights0: list[int]
) -> list[list[int]]:
    weights_full = list()

    for wdelta, winit in zip(weightsd, weights0):
        weights_reconstructed = list()
        for idx, val in enumerate(wdelta):
            weights_reconstructed.append(winit + val)
        weights_full.append(weights_reconstructed)
    return weights_full


@cocotb.test()
@eai_testbench
async def mac_calculation(
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
    period_clk = 10

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

    for data0, gain0, bias0, gaind, initd in zip(
        data_in, weights_in, bias_in, weights_delta, weights_init
    ):
        await RisingEdge(dut.CLK_SYS)

        t0 = get_sim_time("ns")
        dut.DO_CALC.value = 1
        dut.IN_BIAS.value = bias0

        weights_batched = [list(b) for b in batched(gaind, num_mult)]
        data_batched = [list(b) for b in batched(data0, num_mult)]
        for gain, data in zip(weights_batched, data_batched):
            val_data = ""
            for val in data[::-1]:
                val_data += conv_data.integer_to_binary_string_verilog(val).split("b")[
                    -1
                ]
            dut.IN_DATA.value = LogicArray(val_data)
            val_gain = ""
            for val in gain[::-1]:
                val_gain += conv_wght.integer_to_binary_string_verilog(val).split("b")[
                    -1
                ]
                print(val_gain)
            dut.INITIAL_WEIGHT.value = initd
            dut.IN_WEIGHTS.value = LogicArray(val_gain)
            await RisingEdge(dut.CLK_SYS)
            if "gen_delta_consecutive" in dir(dut):
                print("\n")
                val1 = dut.gen_delta_consecutive.weight_decompressed.value
                for val0, weightsd in zip(
                    dut.gen_delta_consecutive.weight_decompressed_comb.value, gain
                ):
                    print(val0.to_signed(), val1.to_signed(), weightsd)

        dut.INITIAL_WEIGHT.value = 0
        dut.IN_WEIGHTS.value = 0
        dut.IN_DATA.value = 0
        for _ in range(2):
            await RisingEdge(dut.CLK_SYS)

        dut.DO_CALC.value = 0
        result = dut.OUT_DATA.value.to_signed()
        t1 = get_sim_time("ns")
        await RisingEdge(dut.CLK_SYS)

        check = model_mac(
            bias=bias0,
            weights=gain0,
            data=data0,
            bitwidth=bitwidth,
            is_signed=is_signed,
        )

        dt = int((t1 - t0) / period_clk)
        assert dt == int(num_params / num_mult) + 2
        if check != result:
            print("\n")
            print(bias0)
            print(data0)
            print(gain0)
            print(initd)
            print(gaind)
            print(check, dut.mac_out.value.to_signed(), dut.OUT_DATA.value.to_signed())
        assert dut.OUT_DATA.value.to_signed() == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, deltawidth", [(8, 4), (10, 6)])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [1, 2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_delta_core_consecutive_with_dsp(
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
    weights_full = get_full_weights_consecutive(
        weightsd=weightsd, weights0=weights0, bitwidth=bitwidth
    )
    cocotb_test_fixture.write(
        {
            "data_in": data0,
            "weights_in": weights_full,
            "bias_in": bias0,
            "weights_delta": weightsd,
            "weights_init": weights0,
        }
    )

    cocotb_test_fixture.set_top_module_name("MAC_DELTA_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(mac_delta, "verilog/mac_delta_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_DELTAWIDTH": deltawidth,
            "NUM_MULT_PARALLEL": num_mult,
            "NUM_SUM_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "DELTA_MODE": 0,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, deltawidth", [(8, 4), (10, 6)])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [1, 2])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_delta_core_referencing_with_dsp(
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
    weights_full = get_full_weights_reference(weightsd=weightsd, weights0=weights0)
    cocotb_test_fixture.write(
        {
            "data_in": data0,
            "weights_in": weights_full,
            "bias_in": bias0,
            "weights_delta": weightsd,
            "weights_init": weights0,
        }
    )

    cocotb_test_fixture.set_top_module_name("MAC_DELTA_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(mac_delta, "verilog/mac_delta_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_DELTAWIDTH": deltawidth,
            "NUM_MULT_PARALLEL": num_mult,
            "NUM_SUM_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "DELTA_MODE": 1,
        },
        defines={},
    )
