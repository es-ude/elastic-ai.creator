import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.testing import CocotbTestFixture, eai_testbench


def adder_model(a: int, b: int, cin: int) -> tuple[int, int]:
    return a ^ b ^ cin, 1 if a + b + cin > 1 else 0


@cocotb.test()
@eai_testbench
async def adder_truthtable(dut):
    input_a = [1, 0, 1, 0, 1, 0, 1, 0]
    input_b = [1, 1, 0, 0, 1, 1, 0, 0]
    input_c = [1, 1, 1, 1, 0, 0, 0, 0]
    for A, B, Cin in zip(input_a, input_b, input_c):
        dut.A.value = A
        dut.B.value = B
        dut.Cin.value = Cin
        await Timer(2, unit="step")
        assert (dut.Q.value, dut.Cout.value) == adder_model(A, B, Cin)


@pytest.mark.simulation
def test_adder_full(cocotb_test_fixture: CocotbTestFixture):
    cocotb_test_fixture.set_top_module_name("ADDER_FULL")
    cocotb_test_fixture.add_srcs_from_package("adders", "verilog/adder_half.v")
    cocotb_test_fixture.run(params={}, defines={})
