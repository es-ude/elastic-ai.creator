import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

import elasticai.creator_plugins.mac_delta as mac_delta
import elasticai.creator_plugins.multipliers as multipliers
from elasticai.creator.arithmetic import int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.mac_delta.tests.mac_delta_core_test import (
    build_testdata,
    get_full_weights_reference,
    model_mac,
)


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
    dut.IN_BRAM.value = 0
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

    conv_data = int_converter(total_bits=bitwidth, signed=is_signed)
    conv_wght = int_converter(total_bits=deltawidth, signed=is_signed)

    for data0, gain0, bias0, gaind, gaini in zip(
        data_in, weights_in, bias_in, weights_delta, weights_init
    ):
        bram_params = list()
        bram_params.extend(gain0)
        bram_params.append(gaini)
        bram_params.append(bias0)
        await RisingEdge(dut.CLK_SYS)

        t0 = get_sim_time("ns")
        dut.DO_CALC.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.DO_CALC.value = 0

        for ite in range(1 + int(num_params / num_mult)):
            idx = split_logical_data_into_list(
                value=dut.IDX_BRAM.value,
                num_values=num_mult,
                bitwidth=dut.INDEX_BITWIDTH.value.to_unsigned(),
            )
            val_data = ""
            val_gain = ""
            # TODO: Fix issue with handling weights initial and bias for processing from BRAM
            for val in idx:
                val_gain += conv_wght.integer_to_binary_string_verilog(
                    bram_params[val]
                ).split("b")[-1]

                if ite > 0:
                    val_data += conv_data.integer_to_binary_string_verilog(
                        data0[val]
                    ).split("b")[-1]
                else:
                    val_data += conv_data.integer_to_binary_string_verilog(0).split(
                        "b"
                    )[-1]
            dut.IN_BRAM.value = LogicArray(val_gain)
            dut.IN_DATA.value = LogicArray(val_data)

            await RisingEdge(dut.CLK_SYS)
            await Timer(1, unit="step")

        await RisingEdge(dut.DATA_RDY)
        t1 = get_sim_time("ns")
        assert dut.DATA_RDY.value == 1
        for _ in range(2):
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
        dt_check = int(num_params / num_mult) + 4
        if dt != dt_check:
            print(dt, dt_check)
        assert dt == dt_check

        if check != result:
            print("\n")
            print(bias0)
            print(data0)
            print(gain0)
            print(bram_params)
            print(
                check,
                dut.MAC_UNIT.mac_out.value.to_signed(),
                dut.OUT_DATA.value.to_signed(),
            )
        assert dut.OUT_DATA.value.to_signed() == check


@pytest.mark.skip
@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, deltawidth",
    [
        (8, 4),
    ],
)
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [1])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_delta_bram_reference_dsp(
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

    cocotb_test_fixture.set_top_module_name("MAC_DELTA")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(mac_delta, "verilog/mac_delta_bram.v")
    cocotb_test_fixture.add_srcs_from_package(mac_delta, "verilog/mac_delta_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_DELTAWIDTH": deltawidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
            "DELTA_MODE": 1,
        },
        defines={},
    )


@pytest.mark.skip
@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, deltawidth", [(12, 4)])
@pytest.mark.parametrize("num_params", [32])
@pytest.mark.parametrize("num_mult", [1])
@pytest.mark.parametrize("is_signed", [True])
def test_mac_delta_bram_reference_dsp_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    deltawidth: int,
    num_params: int,
    num_mult: int,
    is_signed: bool,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    mac_delta.load_and_plugin(
        type="mac_delta_bram",
        id="",
        params={
            "INPUT_BITWIDTH": bitwidth,
            "INPUT_DELTAWIDTH": deltawidth,
            "INPUT_NUM_DATA": num_params,
            "NUM_MULT_PARALLEL": num_mult,
            "DELTA_MODE": 1,
        },
        packages=["mac_delta"],
        path2save=build_dir,
    )
    mac_delta.load_and_plugin(
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
    cocotb_test_fixture.set_top_module_name("MAC_DELTA")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
