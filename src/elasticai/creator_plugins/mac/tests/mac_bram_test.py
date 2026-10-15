from math import ceil, log2

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, ReadOnly, RisingEdge
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

import elasticai.creator_plugins.multipliers as multipliers
from elasticai.creator.arithmetic import int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac import load_and_plugin

from .mac_core_test import build_testdata, model_mac


def split_logical_data_into_list(
    value: str, num_values: int, bitwidth: int
) -> list[int]:
    v = int(value)
    mask = (1 << bitwidth) - 1
    return [(v >> (i * bitwidth)) & mask for i in range(num_values)]


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
    dut.IN_BRAM.value = 0
    dut.IN_DATA.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await ClockCycles(dut.CLK_SYS, 2)
    for idx in range(8):
        dut.RSTN.value = idx % 2
        await ClockCycles(dut.CLK_SYS, 2)
    dut.RSTN.value = 1
    await ClockCycles(dut.CLK_SYS, 2)

    # Apply data and test
    dut.EN.value = 1
    await ClockCycles(dut.CLK_SYS, 2)

    conv = int_converter(total_bits=bitwidth, signed=is_signed)
    for data0, gain0, bias0 in zip(data_in, weights_in, bias_in):
        bram_params = list()
        bram_params.extend(gain0)
        bram_params.append(bias0)
        await RisingEdge(dut.CLK_SYS)

        t0 = get_sim_time("ns")
        dut.DO_CALC.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0

        val_bias = conv.integer_to_binary_string_verilog(bram_params[-1]).split("b")[-1]
        dut.IN_BRAM.value = LogicArray(val_bias)
        dut.IN_DATA.value = conv.integer_to_binary_string_verilog(0).split("b")[-1]
        bram_width = dut.INDEX_BITWIDTH.value.to_unsigned()
        for ite in range(1 + int(num_params / num_mult)):
            await ReadOnly()
            bram_idx = dut.IDX_BRAM.value.to_unsigned()
            idx = split_logical_data_into_list(
                value=bram_idx,
                num_values=num_mult,
                bitwidth=bram_width,
            )
            val_data = ""
            val_gain = ""
            for val in idx:
                val_gain += conv.integer_to_binary_string_verilog(
                    bram_params[val]
                ).split("b")[-1]

                if not bram_idx == len(bram_params) - 1:
                    val_data += conv.integer_to_binary_string_verilog(data0[val]).split(
                        "b"
                    )[-1]
                else:
                    val_data += conv.integer_to_binary_string_verilog(0).split("b")[-1]
            await RisingEdge(dut.CLK_SYS)
            dut.IN_BRAM.value = LogicArray(val_gain)
            dut.IN_DATA.value = LogicArray(val_data)

        await RisingEdge(dut.DATA_RDY)
        assert dut.DATA_RDY.value == 1
        t1 = get_sim_time("ns")
        await ReadOnly()
        result = dut.OUT_DATA.value.to_signed()

        for _ in range(2):
            await RisingEdge(dut.CLK_SYS)
        check = model_mac(
            bias=bias0,
            weights=gain0,
            data=data0,
            bitwidth=bitwidth,
            is_signed=is_signed,
        )

        dt = int((t1 - t0) / period_clk)
        dt_check = int(num_params / num_mult) + 4
        if dt != dt_check:
            print(dt, dt_check)
        assert dt == dt_check

        if check != result:
            print("\n")
            print(f"bias: {bias0}")
            print(f"data: {data0}")
            print(f"weights: {gain0}")
            print(f"bram: {bram_params}")
            print(
                check,
                dut.MAC_UNIT.mac_out.value.to_signed(),
                dut.OUT_DATA.value.to_signed(),
            )
        assert dut.OUT_DATA.value.to_signed() == check


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8])
@pytest.mark.parametrize("num_params", [16, 64])
@pytest.mark.parametrize("num_mult", [1])
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

    cocotb_test_fixture.set_top_module_name("MAC")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_bram.v")
    cocotb_test_fixture.add_srcs_from_package("mac", "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "SIZE_INPUT": num_params,
            "NUM_MULT": num_mult,
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
            "INDEX_BITWIDTH": 1 + int(ceil(log2(num_params))),
            "INDEX_WEIGHTS_START": 0,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [1])
@pytest.mark.parametrize("is_signed", [True])
def test_build_dsp(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mac_bram",
        id="",
        params={
            "BITWIDTH": bitwidth,
            "SIZE_INPUT": num_params,
            "NUM_MULT": num_mult,
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
            "INDEX_BITWIDTH": 1 + int(ceil(log2(num_params))),
            "INDEX_WEIGHTS_START": 0,
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
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [1])
@pytest.mark.parametrize("is_signed", [True])
def test_build_lut(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="mac_bram",
        id="",
        params={
            "BITWIDTH": bitwidth,
            "SIZE_INPUT": num_params,
            "NUM_MULT": num_mult,
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
            "INDEX_BITWIDTH": 1 + int(ceil(log2(num_params))),
            "INDEX_WEIGHTS_START": 0,
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
