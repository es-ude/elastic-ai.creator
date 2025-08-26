import cocotb
import pytest

from elasticai.creator.testing import (
    eai_testbench,
)

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


@cocotb.test()
@eai_testbench
async def my_testbench(dut, x, input_data):
    assert x == dut.X.value
    assert input_data == [1, 2]


@pytest.mark.simulation
@pytest.mark.parametrize(["x"], [(i,) for i in (1, 2, 3)])
def test_my_testbench(cocotb_test_fixture, x):
    additional_input_data = [1, 2]
    cocotb_test_fixture.write({"input_data": additional_input_data})
    cocotb_test_fixture.run(params={"X": x}, defines={})
