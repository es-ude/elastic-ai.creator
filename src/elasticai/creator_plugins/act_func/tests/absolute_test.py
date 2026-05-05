import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


def act_absolute(a: list[int], config: FxpParams) -> list:
    return [abs(val) for val in a]


@cocotb.test()
@eai_testbench
async def check_transfer_function(dut, total_bits: int):
    config = FxpParams(total_bits=total_bits, frac_bits=0, signed=True)
    xstart = config.minimum_as_integer + 1
    xstop = config.maximum_as_integer + 1
    xinput = [xstart + val for val in range(xstop - xstart)]
    xcheck = act_absolute(xinput, config)
    xoutput = []
    for val in xinput:
        dut.A.value = val
        await Timer(2, unit="step")
        xoutput.append(dut.Q.value.to_signed())
    assert xoutput == xcheck


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [4, 8, 12, 16])
def test_absolute(cocotb_test_fixture: CocotbTestFixture, total_bits: int):
    cocotb_test_fixture.set_top_module_name("ACT_ABSOLUTE")
    cocotb_test_fixture.run(params={"BITWIDTH": total_bits}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("total_bits", [6, 8])
def test_absolute_build(cocotb_test_fixture: CocotbTestFixture, total_bits: int):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    id = f"{total_bits:02d}"

    load_and_plugin(
        type="absolute",
        id=id,
        params={"BITWIDTH": total_bits},
        packages=["act_func"],
        path2save=build_dir,
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name(f"ABSOLUTE_{id}")
    cocotb_test_fixture.run(params={}, defines={})
