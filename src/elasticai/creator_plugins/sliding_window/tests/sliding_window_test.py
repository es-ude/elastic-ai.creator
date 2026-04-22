import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray

from elasticai.creator.testing import CocotbTestFixture, eai_testbench

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


@ctb.test()
@eai_testbench
async def check_sliding_window(dut, input_data: str, expected_output: tuple[str]):
    clock = Clock(dut.clk, 10)
    ctb.start_soon(clock.start())
    dut.src_valid.value = 1
    dut.d_in.value = LogicArray(input_data)
    max_read_cycles = 100
    result = []
    dut.dst_ready.value = 1
    for _ in range(max_read_cycles):
        await RisingEdge(dut.clk)

        if dut.valid.value == 1:
            result.append(str(dut.d_out.value))

    dut.dst_ready.value = 0
    assert expected_output == result


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["input_data", "expected_output"],
    [
        (
            "000001010011100101110111",
            ["000", "001", "010", "011", "100", "101", "110", "111"],
        ),
    ],
)
def test_sliding_window(
    cocotb_test_fixture: CocotbTestFixture, input_data: str, expected_output: list[str]
):
    output_width = len(expected_output[0])
    input_width = len(input_data)
    num_steps = len(expected_output)

    stride = (input_width) // num_steps
    print(stride)
    cocotb_test_fixture.run(
        params=dict(INPUT_WIDTH=input_width, OUTPUT_WIDTH=output_width, STRIDE=stride),
        defines={},
    )
