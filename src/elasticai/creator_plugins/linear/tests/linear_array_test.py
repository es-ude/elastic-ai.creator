import cocotb
import numpy as np
import pytest
import torch
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams, int_converter
from elasticai.creator.nn.fixed_point.linear import Linear
from elasticai.creator.testing import CocotbTestFixture, check_results, eai_testbench
from elasticai.creator_plugins import linear, mac, multipliers


def model_linear_layer(
    bitwidth: int, bias: LogicArray, weights: LogicArray, data: list[list[int]]
) -> list[list[int]]:
    params = FxpParams(total_bits=bitwidth, frac_bits=0, signed=True)
    bias0 = [
        LogicArray(str(bias)[i : i + bitwidth]).to_signed()
        for i in range(0, len(bias), bitwidth)
    ]
    num_neurons = len(bias0)
    wght0 = [
        LogicArray(str(weights)[i : i + bitwidth]).to_signed()
        for i in range(0, len(weights), bitwidth)
    ]
    num_weights = int(len(wght0) / num_neurons)
    wght1 = [
        wght0[idx * num_weights : (idx + 1) * num_weights] for idx in range(num_neurons)
    ]

    sum = list()
    for vdata in data:
        psum = list()
        for vbias, vwghts in zip(bias0, wght1):
            vsum = vbias
            for val, wght in zip(vdata, vwghts):
                vsum += val * wght
            psum.append(vsum)
        data_clipped = np.clip(
            np.asarray(psum),
            a_min=params.minimum_as_integer,
            a_max=params.maximum_as_integer,
        ).tolist()
        sum.append(data_clipped)
    return sum


def reconstruct_results(bitwidth: int, result: LogicArray) -> list[int]:
    return [
        LogicArray(str(result)[i : i + bitwidth]).to_signed()
        for i in range(0, len(result), bitwidth)
    ]


@cocotb.test()
@eai_testbench
async def linear_tb(
    dut,
    bitwidth: int,
    num_mult: int,
    size_input: int,
    size_output: int,
    data_in: list[list[int]],
    expected_in: list[list[int]],
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
    if expected_in:
        expected = expected_in
    else:
        expected = model_linear_layer(
            bitwidth=bitwidth,
            bias=dut.BIAS.value,
            weights=dut.WEIGHTS.value,
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
        assert int(num_clocks) == int((size_input * size_output) / num_mult) + 1 + (
            0 if (num_mult == size_input) else 1
        )
        if not check_results(result=result, expected=expected_batch, tol=0):
            print(f"data: {data_batch}")
            print(f"bias: {dut.BIAS.value}")
            print(f"weights: {dut.WEIGHTS.value}")
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
        {
            "data_in": [[4, 3, 2, 4], [3, 2, 4, 4], [2, 4, 4, 3], [4, 4, 3, 2]],
            "expected_in": [[4, -1, -6], [-15, -12, -9], [10, 5, 0]],
        }
    )
    cocotb_test_fixture.set_top_module_name("LINEAR_ARRAY")
    cocotb_test_fixture.clear_srcs()
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
            "expected_in": [],
        }
    )
    cocotb_test_fixture.set_top_module_name("LINEAR_ARRAY_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(glob_pattern="verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, size_input, size_output, num_mult",
    [
        (8, 4, 2, 1),
        (12, 40, 8, 8),
        (16, 512, 64, 8),
    ],
)
def test_codesign(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_mult: int,
    size_input: int,
    size_output: int,
) -> None:

    frac_width = int(np.random.randint(low=1, high=bitwidth, size=1)[0])
    fxp = FxpParams(total_bits=bitwidth, frac_bits=frac_width, signed=True)
    arith = FxpArithmetic(fxp_params=fxp)

    weights = torch.randn(size=(size_output, size_input)) * (
        arith.maximum_as_rational - arith.minimum_as_rational
    )
    weights = torch.clip(
        input=weights / weights.abs().sum(),
        min=fxp.minimum_as_rational,
        max=fxp.maximum_as_rational,
    )
    bias = torch.rand(size=(size_output,)) * (
        arith.maximum_as_rational - arith.minimum_as_rational
    )
    bias = torch.clip(
        input=bias / bias.abs().sum(),
        min=fxp.minimum_as_rational,
        max=fxp.maximum_as_rational,
    )

    dut = Linear(
        in_features=size_input,
        out_features=size_output,
        total_bits=bitwidth,
        frac_bits=frac_width,
    )
    dut.bias = torch.nn.Parameter(arith.round_to_rational(bias))
    dut.weight = torch.nn.Parameter(arith.round_to_rational(weights))
    data_in_int = [
        np.random.randint(
            low=fxp.minimum_as_integer,
            high=fxp.maximum_as_integer,
            size=size_input,
        ).tolist()
        for _ in range(8)
    ]
    data_in_fxp = torch.FloatTensor(data_in_int) * fxp.minimum_step_as_rational

    dut.eval()
    with torch.no_grad():
        expected_out = dut(data_in_fxp) / fxp.minimum_step_as_rational

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut.create_design(name="1").save_to(build_dir, take_vhdl=False, num_mult=num_mult)  # type: ignore
    cocotb_test_fixture.write(
        {
            "data_in": data_in_int,
            "expected_in": expected_out.int().tolist(),
        }
    )
    cocotb_test_fixture.set_top_module_name("LINEAR_ARRAY_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(glob_pattern="verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
