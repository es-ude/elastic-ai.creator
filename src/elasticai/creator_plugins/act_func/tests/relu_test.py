import cocotb
import pytest
from cocotb.triggers import Timer
from torch import asarray

from elasticai.creator.arithmetic import FxpParams
from elasticai.creator.nn.fixed_point import ReLU as layer
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.act_func.utils import load_and_plugin


def act_relu(a: list[int], config: FxpParams) -> list[float]:
    xin = asarray(a)
    lay = layer(total_bits=config.total_bits)
    lay.eval()
    return lay.forward(xin).tolist()


@cocotb.test()
@eai_testbench
async def check_transfer_function(dut, bitwidth: int):
    config = FxpParams(total_bits=bitwidth, frac_bits=0, signed=True)
    xstart = config.minimum_as_integer
    xstop = config.maximum_as_integer + 1
    xrange = [xstart + val for val in range(xstop - xstart)]
    xoutput = []
    xcheck = act_relu(xrange, config)
    for val in xrange:
        stimulus = val
        dut.A.value = stimulus
        await Timer(2, unit="step")
        xoutput.append(dut.Q.value.to_signed())
    assert xoutput == xcheck


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [4, 8, 12, 16])
def test_relu(cocotb_test_fixture: CocotbTestFixture, bitwidth: int):
    cocotb_test_fixture.set_top_module_name("ACT_RELU")
    cocotb_test_fixture.run(params={"BITWIDTH": bitwidth}, defines={})


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8, 10])
def test_relu_build(cocotb_test_fixture: CocotbTestFixture, bitwidth: int):
    build_dir = cocotb_test_fixture.get_artifact_dir() / "verilog"
    id = f"{bitwidth}"

    load_and_plugin(
        type="relu",
        id=id,
        params={"BITWIDTH": bitwidth},
        packages=["act_func"],
        path2save=build_dir,
    )

    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.set_top_module_name(f"RELU_{id}")
    cocotb_test_fixture.run(params={}, defines={})
