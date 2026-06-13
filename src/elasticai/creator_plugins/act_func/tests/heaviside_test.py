import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


def act_heaviside(xin: list[float], config: FxpParams) -> list:
    return [1.0 if val >= 0 else 0.0 for val in xin]


@cocotb.test()
@eai_testbench
async def check_transfer_function(dut, total_bits: int, frac_bits: int):
    config = FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    xstart = config.minimum_as_integer
    xstop = config.maximum_as_integer + 1
    xinput = [xstart + val for val in range(xstop - xstart)]
    xfloat = [
        (xstart + val) * config.minimum_step_as_rational
        for val in range(xstop - xstart)
    ]
    xcheck = act_heaviside(xfloat, config)
    xoutput = []
    for val in xinput:
        dut.A.value = val
        await Timer(2, unit="step")
        xoutput.append(dut.Q.value.to_signed() * config.minimum_step_as_rational)

    assert xoutput == xcheck


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [4])
@pytest.mark.parametrize("frac_bits", [2])
def test_heaviside(
    cocotb_test_fixture: CocotbTestFixture, total_bits: int, frac_bits: int
):
    cocotb_test_fixture.set_top_module_name("ACT_HEAVISIDE")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [6, 8])
@pytest.mark.parametrize("frac_bits", [3, 4])
def test_heaviside_build(
    cocotb_test_fixture: CocotbTestFixture, total_bits: int, frac_bits: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    id = f"{total_bits:02d}_{frac_bits:02d}"

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    max_val = arith.clamp(arith.cut_as_integer(1.0))
    min_val = arith.clamp(arith.cut_as_integer(0.0))

    load_and_plugin(
        type="heaviside",
        id=id,
        params={"BITWIDTH": total_bits, "MAX_VAL": max_val, "MIN_VAL": min_val},
        packages=["act_func"],
        path2save=build_dir,
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name(f"HEAVISIDE_{id}")
    cocotb_test_fixture.run(params={}, defines={})
