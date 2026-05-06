import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.testing import CocotbTestFixture, eai_testbench


def adder_model(a: int, b: int) -> tuple[int, int]:
    return a ^ b, 1 if a + b > 1 else 0


@cocotb.test()
@eai_testbench
async def adder_truthtable(dut):
    input_a = [1, 0, 1, 0]
    input_b = [1, 1, 0, 0]
    for A, B in zip(input_a, input_b):
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit="step")
        assert (dut.Q.value, dut.Cout.value) == adder_model(A, B)


@pytest.mark.simulation
def test_adder_half(cocotb_test_fixture: CocotbTestFixture):
    cocotb_test_fixture.set_top_module_name("ADDER_HALF")
    cocotb_test_fixture.run(params={}, defines={})
