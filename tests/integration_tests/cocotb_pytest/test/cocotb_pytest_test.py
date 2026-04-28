from importlib.machinery import ModuleSpec
from typing import cast

import cocotb
import pytest

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import CocotbTestFixture, eai_testbench


@cocotb.test()
@eai_testbench
async def my_testbench(dut, x, input_data):
    assert x == dut.X.value
    assert input_data == [1, 2]


@pytest.mark.simulation
@pytest.mark.parametrize(["x"], [(i,) for i in (1, 2, 3)])
def test_can_store_parameters_in_artifact_dir(
    cocotb_test_fixture: CocotbTestFixture, x
):
    additional_input_data = [1, 2]
    cocotb_test_fixture.write({"input_data": additional_input_data})
    with get_file_from_package(
        cast(str, cast(ModuleSpec, __spec__).parent), "../vhdl/my_testbench.vhd"
    ) as p:
        cocotb_test_fixture.set_srcs((p.absolute(),))
        cocotb_test_fixture.set_top_module_name("my_testbench")
        cocotb_test_fixture.run(params={"X": x}, defines={})


@pytest.mark.simulation
def test_can_add_srcs_from_package(cocotb_test_fixture: CocotbTestFixture):
    x = 1
    cocotb_test_fixture.write({"input_data": [1, 2], "x": x})
    package = ".".join(__spec__.parent.split(".")[0:-1])
    print(package)
    cocotb_test_fixture.add_srcs_from_package(package, "vhdl/*.vhd")
