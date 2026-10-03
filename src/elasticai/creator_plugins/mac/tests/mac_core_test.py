from itertools import batched
from random import randint

import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

import elasticai.creator_plugins.adders as adders
import elasticai.creator_plugins.multipliers as multipliers
from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench


def build_testdata(
    bitwidth: int, num_params: int, is_signed: bool, repeats: int = 16
) -> tuple[list[list[int]], list[list[int]], list[int]]:
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)

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
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
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
    dut.DO_CLEAR.value = 0
    dut.IN_BIAS.value = 0
    dut.IN_WEIGHTS.value = 0
    dut.IN_DATA.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await ClockCycles(dut.CLK_SYS, 4)
    for idx in range(8):
        dut.RSTN.value = idx % 2
        await ClockCycles(dut.CLK_SYS, 2)
    dut.RSTN.value = 1
    await ClockCycles(dut.CLK_SYS, 4)
    dut.EN.value = 1
    await ClockCycles(dut.CLK_SYS, 4)
    conv = int_converter(total_bits=bitwidth, signed=is_signed)

    for data0, gain0, bias0 in zip(data_in, weights_in, bias_in):
        t0 = get_sim_time("ns")
        dut.DO_CALC.value = 1
        dut.DO_CLEAR.value = 1
        dut.IN_BIAS.value = bias0
        weights_batched = [list(b) for b in batched(data0, num_mult)]
        data_batched = [list(b) for b in batched(gain0, num_mult)]
        for gain, data in zip(weights_batched, data_batched):
            dut.IN_DATA.value = LogicArray(
                "".join(
                    conv.integer_to_binary_string_verilog(d).split("b")[-1]
                    for d in data
                )
            )
            dut.IN_WEIGHTS.value = LogicArray(
                "".join(
                    conv.integer_to_binary_string_verilog(g).split("b")[-1]
                    for g in gain
                )
            )
            await RisingEdge(dut.CLK_SYS)
            dut.DO_CLEAR.value = 0

        dut.IN_WEIGHTS.value = 0
        dut.IN_DATA.value = 0
        await ClockCycles(dut.CLK_SYS, 1)
        dut.DO_CALC.value = 0
        await ReadOnly()
        t1 = get_sim_time("ns")
        result = dut.OUT_DATA.value.to_signed()
        await ClockCycles(dut.CLK_SYS, 4)
        # Checking results
        check = model_mac(
            bias=bias0,
            weights=gain0,
            data=data0,
            bitwidth=bitwidth,
            is_signed=is_signed,
        )
        dt = int((t1 - t0) / period_clk)
        assert dt == int(num_params / num_mult) + 1
        if check != result:
            print("\n")
            print(bias0)
            print(data0)
            print(gain0)
            print(
                check,
                dut.mac_out.value.to_signed(),
                dut.OUT_DATA.value.to_signed(),
                result,
            )
        assert result == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_clamp_overflow(
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

    cocotb_test_fixture.set_top_module_name("MAC_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "BITS_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
            "NUM_MULT": num_mult,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [2])
@pytest.mark.parametrize("is_signed", [True])
def test_clamp_underflow(
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

    cocotb_test_fixture.set_top_module_name("MAC_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "BITS_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
            "NUM_MULT": num_mult,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8, 6])
@pytest.mark.parametrize("num_params", [8, 32])
@pytest.mark.parametrize("num_mult", [1, 4])
@pytest.mark.parametrize("is_signed", [True])
def test_template_dsp(
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

    cocotb_test_fixture.set_top_module_name("MAC_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "BITS_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [6, 8])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [4])
@pytest.mark.parametrize("is_signed", [True])
def test_template_lut(
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

    cocotb_test_fixture.set_top_module_name("MAC_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_lut_signed.v")
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "BITS_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [6, 8])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [4])
@pytest.mark.parametrize("is_signed", [True])
def test_template_dadda(
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

    cocotb_test_fixture.set_top_module_name("MAC_CORE")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(
        multipliers, f"verilog/mult_dadda_s{bitwidth}.v"
    )
    cocotb_test_fixture.add_srcs_from_package(adders, "verilog/adder_*.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "BITS_OVERSIZE": int(np.ceil(np.log2(num_params))),
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
        },
        defines={},
    )
