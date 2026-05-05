import cocotb
import pytest
from cocotb.triggers import Timer
from torch import asarray

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn.fixed_point import hard_tanh as layer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


def act_hardtanhsign(a: list[float], config: FxpParams) -> list:
    xin = asarray(a)
    lay = layer.HardTanh(total_bits=config.total_bits, frac_bits=config.frac_bits)
    lay.eval()
    return lay.forward(xin).tolist()


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
    xcheck = act_hardtanhsign(xfloat, config)
    xoutput = []
    for val in xinput:
        dut.A.value = val
        await Timer(2, unit="step")
        xoutput.append(dut.Q.value.to_signed() * config.minimum_step_as_rational)
    assert xoutput == xcheck


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [4])
@pytest.mark.parametrize("frac_bits", [2])
def test_hardtanh(
    cocotb_test_fixture: CocotbTestFixture, total_bits: int, frac_bits: int
):
    cocotb_test_fixture.set_top_module_name("ACT_HARDTANH")
    cocotb_test_fixture.run(params={}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [4, 8])
@pytest.mark.parametrize("frac_bits", [3])
def test_hardtanh_build(
    cocotb_test_fixture: CocotbTestFixture, total_bits: int, frac_bits: int
):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    id = f"{total_bits:02d}_{frac_bits:02d}"

    arith = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )
    max_val = arith.clamp(arith.cut_as_integer(1.0))
    min_val = arith.clamp(arith.cut_as_integer(-1.0))

    load_and_plugin(
        type="hardtanh",
        id=id,
        params={"BITWIDTH": total_bits, "MAX_VAL": max_val, "MIN_VAL": min_val},
        packages=["act_func"],
        path2save=build_dir,
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name(f"HARDTANH_{id}")
    cocotb_test_fixture.run(params={}, defines={})
