import cocotb
import pytest
from cocotb.triggers import Timer
from torch import asarray

import elasticai.creator.nn.fixed_point as nn
from elasticai.creator.arithmetic import FxpArithmetic, FxpConverter, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def precomputed_transfer_func(
    dut,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
    input: list,
    check: list,
    allowed_mae: float,
):
    assert len(input) == len(check)

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    xinput = [
        idx
        for idx in range(
            arith.config.minimum_as_integer, arith.config.maximum_as_integer + 1
        )
    ]

    xoutput = []
    for xin in xinput:
        dut.A.value = xin
        await Timer(2, unit="step")
        xoutput.append(dut.Q.value.to_signed())
    if not xoutput == check:
        print("xin: ", xinput)
        print("xout: ", xoutput)
        print("ref: ", check)

        abs_diff = sum([abs(val0 - val1) for val0, val1 in zip(xoutput, check)]) / len(
            xoutput
        )
        assert abs_diff < allowed_mae
    else:
        assert xoutput == check

    assert xinput == input


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [4])
@pytest.mark.parametrize("frac_bits", [3])
@pytest.mark.parametrize("num_steps", [8])
def test_precomputed(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    cocotb_test_fixture.write(
        {
            "input": [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
            "check": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        }
    )
    cocotb_test_fixture.set_top_module_name("ACT_PRECOMPUTED")
    cocotb_test_fixture.write({"allowed_mae": 0})
    cocotb_test_fixture.run(params={"BITWIDTH": 4}, defines={})


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [6, 8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("num_steps", [16, 32])
def test_precomputed_build_tanh(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    dut = nn.Tanh(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps)
    dut.eval()
    data = dut.get_lut_integer()

    id = f"tanh_{total_bits:02d}_{frac_bits:02d}_{num_steps:02d}"
    cnv = FxpConverter(FxpParams(total_bits=total_bits, frac_bits=0, signed=True))
    ref_str = (
        "{ "
        + ", ".join([cnv.integer_to_binary_string_verilog(val) for val in data[1]])
        + " }"
    )

    load_and_plugin(
        type="precomputed",
        id=id,
        params={
            "BITWIDTH": total_bits,
            "NUM_VALUES": len(data[0]),
            "PRECOMPUTED": ref_str,
        },
        packages=["act_func"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
    )

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    xinput = [
        val for val in range(arith.minimum_as_integer, arith.maximum_as_integer + 1)
    ]
    xcheck = (
        dut(asarray(xinput) * arith.config.minimum_step_as_rational)
        / arith.config.minimum_step_as_rational
    )
    xcheck = xcheck.int().tolist()

    cocotb_test_fixture.write({"input": xinput, "check": xcheck, "allowed_mae": 2})
    cocotb_test_fixture.set_top_module_name(f"PRECOMPUTED_{id.upper()}")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [6, 8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("num_steps", [16, 32])
def test_precomputed_build_silu(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    dut = nn.SiLU(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps)
    dut.eval()
    data = dut.get_lut_integer()

    id = f"tanh_{total_bits:02d}_{frac_bits:02d}_{num_steps:02d}"
    cnv = FxpConverter(FxpParams(total_bits=total_bits, frac_bits=0, signed=True))
    ref_str = (
        "{ "
        + ", ".join([cnv.integer_to_binary_string_verilog(val) for val in data[1]])
        + " }"
    )

    load_and_plugin(
        type="precomputed",
        id=id,
        params={
            "BITWIDTH": total_bits,
            "NUM_VALUES": len(data[0]),
            "PRECOMPUTED": ref_str,
        },
        packages=["act_func"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
    )

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    xinput = [
        val for val in range(arith.minimum_as_integer, arith.maximum_as_integer + 1)
    ]
    xcheck = (
        dut(asarray(xinput) * arith.config.minimum_step_as_rational)
        / arith.config.minimum_step_as_rational
    )
    xcheck = xcheck.int().tolist()

    cocotb_test_fixture.write({"input": xinput, "check": xcheck, "allowed_mae": 2.5})
    cocotb_test_fixture.set_top_module_name(f"PRECOMPUTED_{id.upper()}")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [6, 8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("num_steps", [16, 32])
def test_precomputed_build_sigmoid(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    dut = nn.Sigmoid(total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps)
    dut.eval()
    data = dut.get_lut_integer()

    id = f"tanh_{total_bits:02d}_{frac_bits:02d}_{num_steps:02d}"
    cnv = FxpConverter(FxpParams(total_bits=total_bits, frac_bits=0, signed=True))
    ref_str = (
        "{ "
        + ", ".join([cnv.integer_to_binary_string_verilog(val) for val in data[1]])
        + " }"
    )

    load_and_plugin(
        type="precomputed",
        id=id,
        params={
            "BITWIDTH": total_bits,
            "NUM_VALUES": len(data[0]),
            "PRECOMPUTED": ref_str,
        },
        packages=["act_func"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
    )

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    xinput = [
        val for val in range(arith.minimum_as_integer, arith.maximum_as_integer + 1)
    ]
    xcheck = (
        dut(asarray(xinput) * arith.config.minimum_step_as_rational)
        / arith.config.minimum_step_as_rational
    )
    xcheck = xcheck.int().tolist()

    cocotb_test_fixture.write({"input": xinput, "check": xcheck, "allowed_mae": 2})
    cocotb_test_fixture.set_top_module_name(f"PRECOMPUTED_{id.upper()}")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [6, 8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("num_steps", [16, 32])
def test_precomputed_build_adapt_silu(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    dut = nn.AdaptableSiLU(
        total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps
    )
    dut.eval()
    data = dut.get_lut_integer()

    id = f"tanh_{total_bits:02d}_{frac_bits:02d}_{num_steps:02d}"
    cnv = FxpConverter(FxpParams(total_bits=total_bits, frac_bits=0, signed=True))
    ref_str = (
        "{ "
        + ", ".join([cnv.integer_to_binary_string_verilog(val) for val in data[1]])
        + " }"
    )

    load_and_plugin(
        type="precomputed",
        id=id,
        params={
            "BITWIDTH": total_bits,
            "NUM_VALUES": len(data[0]),
            "PRECOMPUTED": ref_str,
        },
        packages=["act_func"],
        path2save=cocotb_test_fixture.get_artifact_dir() / "verilog",
    )

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    xinput = [
        val for val in range(arith.minimum_as_integer, arith.maximum_as_integer + 1)
    ]
    xcheck = (
        dut(asarray(xinput) * arith.config.minimum_step_as_rational)
        / arith.config.minimum_step_as_rational
    )
    xcheck = xcheck.int().tolist()

    cocotb_test_fixture.write({"input": xinput, "check": xcheck, "allowed_mae": 2.5})
    cocotb_test_fixture.set_top_module_name(f"PRECOMPUTED_{id.upper()}")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
