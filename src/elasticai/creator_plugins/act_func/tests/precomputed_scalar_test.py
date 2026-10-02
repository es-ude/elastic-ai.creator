import cocotb
import pytest
from cocotb.triggers import Timer
from matplotlib import pyplot as plt
from torch import asarray, no_grad

import elasticai.creator.nn.fixed_point as nn
from elasticai.creator.arithmetic import FxpArithmetic, FxpConverter, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


def plot_data(din: list[int], dout: list[int], xref: list[int]) -> None:
    plt.figure()
    plt.plot(din, dout, "r", label="HW")
    plt.plot(din, xref, "k", label="SW")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


@cocotb.test()
@eai_testbench
async def precomputed_func(
    dut,
    func: str,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
    input: list,
    check: list,
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
        plot_data(din=xinput, dout=xoutput, xref=check)
    assert xoutput == check
    assert xinput == input


@pytest.mark.simulation
@pytest.mark.parametrize("func", ["template"])
@pytest.mark.parametrize("total_bits", [4])
@pytest.mark.parametrize("frac_bits", [3])
@pytest.mark.parametrize("num_steps", [8])
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    func: str,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    cocotb_test_fixture.write(
        {
            "input": [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
            "check": [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(
        "act_func", "verilog/precomputed_scalar.v"
    )
    cocotb_test_fixture.set_top_module_name("ACT_PRECOMPUTED_SCALAR")
    cocotb_test_fixture.run(params={"BITWIDTH_IN": 4, "BITWIDTH_OUT": 4}, defines={})


FUNC_REGISTRY = {
    "Tanh": nn.Tanh,
    "Sigmoid": nn.Sigmoid,
    "SiLU": nn.SiLU,
}


@pytest.mark.slow
@pytest.mark.simulation
@pytest.mark.parametrize("func", list(FUNC_REGISTRY.keys()))
@pytest.mark.parametrize("total_bits, frac_bits, num_steps", [(6, 5, 16), (8, 5, 64)])
def test_build(
    cocotb_test_fixture: CocotbTestFixture,
    func: str,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):
    dut = FUNC_REGISTRY[func](
        total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps, use_lut=False
    )
    dut.eval()
    data = dut.get_lut_integer()

    id = f"{func.lower()}_{total_bits:02d}_{frac_bits:02d}_{num_steps:02d}"
    cnv = FxpConverter(FxpParams(total_bits=total_bits, frac_bits=0, signed=True))

    load_and_plugin(
        type="precomputed_scalar",
        id=id,
        params={
            "BITWIDTH_IN": total_bits,
            "BITWIDTH_OUT": total_bits,
            "NUM_VALUES": len(data[0]),
            "VALUES_X": cnv.integer_to_decimal_string_array_verilog(data[0][::-1]),
            "VALUES_Y": cnv.integer_to_decimal_string_array_verilog(data[1][::-1]),
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

    cocotb_test_fixture.write({"input": xinput, "check": xcheck})
    cocotb_test_fixture.set_top_module_name(f"PRECOMPUTED_SCALAR_{id.upper()}")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("func", list(FUNC_REGISTRY.keys()))
@pytest.mark.parametrize("total_bits", [8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("num_steps", [16, 32])
def test_codesign(
    cocotb_test_fixture: CocotbTestFixture,
    func: str,
    total_bits: int,
    frac_bits: int,
    num_steps: int,
):

    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    dut = FUNC_REGISTRY[func](
        total_bits=total_bits, frac_bits=frac_bits, num_steps=num_steps, use_lut=False
    )
    dut.create_design("1").save_to(destination=build_dir, take_vhdl=False)  # type: ignore

    files_available = [file.name for file in build_dir.glob("*.v")]
    files_available.sort()
    assert files_available == ["precomputed_scalar_1.v"]

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    xinput = [
        val for val in range(arith.minimum_as_integer, arith.maximum_as_integer + 1)
    ]
    dut.eval()
    with no_grad():
        xcheck = (
            dut(asarray(xinput) * arith.config.minimum_step_as_rational)
            / arith.config.minimum_step_as_rational
        )
        xcheck = xcheck.int().tolist()

    cocotb_test_fixture.write({"input": xinput, "check": xcheck})
    cocotb_test_fixture.set_top_module_name("PRECOMPUTED_SCALAR_1")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir(glob_pattern="verilog/*.v")
    cocotb_test_fixture.run(
        params={},
        defines={},
    )
