import cocotb
import numpy as np
import pytest
import torch
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge
from cocotb.types import LogicArray
from cocotb.utils import get_sim_time

from elasticai.creator.arithmetic import FxpParams, int_converter
from elasticai.creator.nn.fixed_point.linear import BatchNormedLinear
from elasticai.creator.testing import CocotbTestFixture, check_results, eai_testbench

from .linear_array_test import model_linear_layer, reconstruct_results


@cocotb.test()
@eai_testbench
async def batchnorm_linear_tb(
    dut,
    bitwidth: int,
    fracwidth: int,
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
    right_cases = 0
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
        if not check_results(result=result, expected=expected_batch, tol=1):
            print(f"data: {data_batch}")
            print(f"bias: {dut.BIAS.value}")
            print(f"weights: {dut.WEIGHTS.value}")
            print(f"result: {result}")
            print(f"expected: {expected_batch}")
        else:
            right_cases += 1
    await ClockCycles(dut.CLK_SYS, 4)
    assert right_cases / len(data_in) >= 1.0


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, fracwidth, size_input, size_output, num_mult",
    [
        (8, 1, 4, 2, 1),
        (8, 3, 8, 2, 1),
        (12, 5, 40, 8, 8),
        (16, 8, 128, 64, 8),
    ],
)
def test_codesign(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    fracwidth: int,
    num_mult: int,
    size_input: int,
    size_output: int,
) -> None:

    fxp = FxpParams(total_bits=bitwidth - 2, frac_bits=fracwidth, signed=True)
    dut = BatchNormedLinear(
        in_features=size_input,
        out_features=size_output,
        total_bits=bitwidth,
        frac_bits=fracwidth,
        bn_affine=True,
    )

    data_in_int = [
        np.random.randint(
            low=int(fxp.minimum_as_integer / 2),
            high=int((fxp.maximum_as_integer - 1) / 2),
            size=size_input,
        ).tolist()
        for _ in range(8)
    ]
    fxp = FxpParams(total_bits=bitwidth, frac_bits=fracwidth, signed=True)
    data_in_fxp = torch.FloatTensor(data_in_int) * fxp.minimum_step_as_rational

    dut.eval()
    with torch.no_grad():
        expected_out = dut(data_in_fxp) / fxp.minimum_step_as_rational

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut.create_design(name="1").save_to(build_dir, take_vhdl=False, num_mult=num_mult)  # type: ignore
    files_available = [file.name for file in build_dir.glob("*.v")]
    files_available.sort()
    assert files_available == [
        "linear_array_1.v",
        "mac_array.v",
        "mac_core.v",
        "mult_dsp_signed.v",
    ]

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
