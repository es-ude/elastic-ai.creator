import cocotb
import pytest
from cocotb.triggers import Timer
from torch import asarray

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn.fixed_point import PReLU
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


def act_prelu(a: list[float], scaling: float, config: FxpParams) -> list:
    lay = PReLU(
        total_bits=config.total_bits,
        frac_bits=config.frac_bits,
        num_parameters=1,
        init=scaling,
    )
    lay.eval()

    xin = asarray(a)
    steps = lay.forward(xin)
    return steps.tolist()


@cocotb.test()
@eai_testbench
async def check_transfer_function(dut, total_bits: int, frac_bits: int, scaling: float):
    config = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    xstart = config.minimum_as_integer
    xstop = config.maximum_as_integer + 1
    xinput = [xstart + val for val in range(xstop - xstart)]
    xfloat = [
        (xstart + val) * config.minimum_step_as_rational
        for val in range(xstop - xstart)
    ]
    xcheck = act_prelu(xfloat, scaling, config)
    xoutput = []
    for val in xinput:
        dut.A.value = val
        await Timer(2, unit="step")
        xoutput.append(dut.Q.value.to_signed() * config.minimum_step_as_rational)
    if not xoutput == xcheck:
        print("xinput:", xfloat)
        print("xoutput:", xoutput)
        print("xcheck:", xcheck)
    assert xoutput == xcheck


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [4])
@pytest.mark.parametrize("frac_bits", [2])
@pytest.mark.parametrize("scaling", [0.5])
def test_prelu(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    scaling: float,
):
    cocotb_test_fixture.set_top_module_name("ACT_PRELU")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("scaling", [0.34375, 0.25, 0.125])
def test_build(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    scaling: float,
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    id = f"{total_bits:02d}_{frac_bits:02d}"

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    scale = arith.cut_as_integer(scaling)

    load_and_plugin(
        type="prelu",
        id=id,
        params={
            "BITWIDTH_IN": total_bits,
            "BITWIDTH_OUT": total_bits,
            "FRACWIDTH": frac_bits,
            "SCALING": scale,
        },
        packages=["act_func"],
        path2save=build_dir,
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name(f"PRELU_{id}")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [8])
@pytest.mark.parametrize("frac_bits", [5])
@pytest.mark.parametrize("scaling", [0.125])
def test_codesign(
    cocotb_test_fixture: CocotbTestFixture,
    total_bits: int,
    frac_bits: int,
    scaling: float,
) -> None:
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"

    dut = PReLU(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_parameters=1,
        init=scaling,
    )
    dut.create_design(name="1").save_to(build_dir, take_vhdl=False)  # type: ignore

    files_available = [file.name for file in build_dir.glob("*.v")]
    files_available.sort()
    assert files_available == [
        "prelu_1.v",
    ]

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name("PRELU_1")
    cocotb_test_fixture.run(params={}, defines={})
