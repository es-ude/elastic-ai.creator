import cocotb
import numpy as np
import pytest
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

from elasticai.creator.arithmetic import FxpParams, int_converter
from elasticai.creator.testing import CocotbTestFixture, check_results, eai_testbench
from elasticai.creator_plugins import linear, mac, multipliers

from .linear_array_test import model_linear_layer, reconstruct_results


@cocotb.test()
@eai_testbench
async def linear_tb(
    dut,
    bitwidth: int,
    num_mult: int,
    size_input: int,
    size_output: int,
    data_in: list[list[int]],
):
    period_clk = 2
    conv = int_converter(total_bits=bitwidth, signed=True)

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.EN.value = 0
    dut.DO_CALC.value = 0
    dut.DATA_IN.value = 0

    # Start clock and make reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    await ClockCycles(dut.CLK_SYS, 4)
    for idx in range(8):
        dut.RSTN.value = idx % 2
        await ClockCycles(dut.CLK_SYS, 2)
    dut.RSTN.value = 1

    await ClockCycles(dut.CLK_SYS, 2)
    assert dut.DATA_VALID.value == 1
    dut.EN.value = 1

    await ClockCycles(dut.CLK_SYS, 4)
    expected = model_linear_layer(
        bitwidth=bitwidth,
        bias=dut.LINEAR.BIAS.value,
        weights=dut.LINEAR.WEIGHTS.value,
        data=data_in,
    )
    for data_batch, expected_batch in zip(data_in, expected):
        dut.DO_CALC.value = 1
        dut.DATA_IN.value = LogicArray(
            "".join(
                conv.integer_to_binary_string_verilog(d).split("b")[-1]
                for d in data_batch
            )
        )
        await ClockCycles(dut.CLK_SYS, 2)
        dut.DO_CALC.value = 0

        await FallingEdge(dut.DATA_VALID)
        t0 = get_sim_time(unit="ns")
        await RisingEdge(dut.DATA_VALID)
        t1 = get_sim_time(unit="ns")
        await ReadOnly()
        await ClockCycles(dut.CLK_SYS, 2)
        result = reconstruct_results(bitwidth=bitwidth, result=dut.DATA_OUT.value)
        num_clocks = (t1 - t0) / period_clk
        await ClockCycles(dut.CLK_SYS, 8)
        # Check results
        assert int(num_clocks) == int((size_input * size_output) / num_mult) + 3
        passed = check_results(result=result, expected=expected_batch, tol=1)
        if not passed:
            print(f"data: {data_batch}")
            print(f"bias: {dut.LINEAR.BIAS.value}")
            print(f"weights: {dut.LINEAR.WEIGHTS.value}")
            print(f"result: {result}")
            print(f"expected: {expected_batch}")
            assert False
    await ClockCycles(dut.CLK_SYS, 4)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, size_input, size_output, num_mult",
    [
        (8, 4, 3, 1),
        (8, 4, 3, 2),
        (8, 4, 3, 4),
    ],
)
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_mult: int,
    size_input: int,
    size_output: int,
) -> None:

    cocotb_test_fixture.write(
        {"data_in": [[4, 3, 2, 4], [3, 2, 4, 4], [2, 4, 4, 3], [4, 4, 3, 2]]}
    )
    cocotb_test_fixture.set_top_module_name("LINEAR_ARRAY_PIPELINED")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(
        linear, "verilog/linear_array_pipelined.v"
    )
    cocotb_test_fixture.add_srcs_from_package(linear, "verilog/linear_array.v")
    cocotb_test_fixture.add_srcs_from_package(mac, "verilog/mac_array.v")
    cocotb_test_fixture.add_srcs_from_package(mac, "verilog/mac_core.v")
    cocotb_test_fixture.add_srcs_from_package(multipliers, "verilog/mult_dsp_signed.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "SIZE_INPUT": size_input,
            "SIZE_OUTPUT": size_output,
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, size_input, size_output, num_mult",
    [
        (8, 4, 2, 1),
        (8, 6, 3, 1),
        (8, 8, 2, 2),
    ],
)
def test_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_mult: int,
    size_input: int,
    size_output: int,
) -> None:
    from elasticai.creator_plugins.linear.utils import load_and_plugin

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    fxp = FxpParams(total_bits=bitwidth, frac_bits=0, signed=True)
    conv = int_converter(total_bits=bitwidth, signed=True)
    load_and_plugin(
        type="linear_array_pipelined",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "SIZE_INPUT": size_input,
            "SIZE_OUTPUT": size_output,
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
        },
        packages=["linear"],
        path2save=build_dir,
    )
    load_and_plugin(
        type="linear_array",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "NUM_MULT": num_mult,
            "SIZE_INPUT": size_input,
            "SIZE_OUTPUT": size_output,
            "BITS_SCALE_BIAS": 0,
            "BITS_SCALE_DOUT": 0,
            "WEIGHTS": conv.integer_to_decimal_string_array_verilog(
                np.random.randint(
                    low=fxp.minimum_as_integer,
                    high=fxp.maximum_as_integer,
                    size=size_input * size_output,
                ).tolist()
            ),
            "BIAS": conv.integer_to_decimal_string_array_verilog(
                np.random.randint(
                    low=fxp.minimum_as_integer,
                    high=fxp.maximum_as_integer,
                    size=size_output,
                ).tolist()
            ),
        },
        packages=["linear"],
        path2save=build_dir,
    )
    files_available = [file.name for file in build_dir.glob("*.v")]
    files_available.sort()
    assert files_available == [
        "linear_array_0.v",
        "linear_array_pipelined_0.v",
        "mac_array.v",
        "mac_core.v",
        "mult_dsp_signed.v",
    ]

    cocotb_test_fixture.write(
        {
            "data_in": [
                np.random.randint(
                    low=fxp.minimum_as_integer,
                    high=fxp.maximum_as_integer,
                    size=size_input,
                ).tolist()
                for _ in range(4)
            ],
        }
    )
    cocotb_test_fixture.set_top_module_name("LINEAR_ARRAY_PIPELINED_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(glob_pattern="verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
