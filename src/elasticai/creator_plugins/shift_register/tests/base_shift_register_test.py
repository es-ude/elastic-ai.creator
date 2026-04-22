import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray, Range

from elasticai.creator.testing import eai_testbench

from .helpers import _reset_dut, _write_current_d_in

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"

# Clock period in nanoseconds
CLOCK_PERIOD_NS = 10


def _to_logic_array(value: int, width: int) -> LogicArray:
    max_value = (1 << width) - 1
    if value > max_value:
        raise ValueError(
            f"Value {value} exceeds maximum for {width}-bit width (max: {max_value})"
        )
    return LogicArray(value, Range(width - 1, "downto", 0))


async def _write_data(dut, value: int, data_width: int) -> None:
    dut.d_in.value = _to_logic_array(value, data_width)
    await _write_current_d_in(dut)


async def _write_array(dut, values: list[int], data_width: int) -> None:
    """write to shift register by producing rising edges on src_valid, just like upstream clients would do."""
    for v in values:
        await _write_data(dut, v, data_width)


async def _record_output_for_n_cycles(
    dut, num_points, data_width, n
) -> list[list[int]]:
    """read the shift register when valid is high, just like downstream clients would do"""
    values = []

    def get_ouput_as_ints() -> list[int]:
        sliced: list[LogicArray] = [
            dut.d_out.value[(i + 1) * data_width - 1 : i * data_width]
            for i in reversed(range(num_points))
        ]
        return [s.to_unsigned() for s in sliced]

    for _ in range(n):
        await RisingEdge(dut.clk)
        # Capture output on rising edge of valid (0->1 transition)
        if dut.valid.value == 1:
            values.append(get_ouput_as_ints())
    return values


@ctb.test()
@eai_testbench
async def shift_register_accumulates_data(
    dut,
    data_width: int,
    num_points: int,
    input_data: list[int],
    expected_output: list[list[int]],
):
    clock = Clock(dut.clk, CLOCK_PERIOD_NS)
    ctb.start_soon(clock.start())
    await _reset_dut(dut)
    dut.dst_ready.value = 1
    # we need two cycles per data write and one cycle for valid signal propagation
    num_expected_read_cycles = len(input_data) + 1

    read = ctb.start_soon(
        _record_output_for_n_cycles(
            dut, num_points, data_width, num_expected_read_cycles
        )
    )
    await _write_array(dut, input_data, data_width)
    await read

    actual_output = read.result()
    assert expected_output == actual_output


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["data_width", "num_points", "input_data", "expected_output"],
    [
        (4, 4, [1, 2, 3, 4, 5, 6], [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]),
        (3, 3, [2, 5, 7, 6, 3], [[2, 5, 7], [5, 7, 6], [7, 6, 3]]),
        (5, 2, [12, 23, 0, 4], [[12, 23], [23, 0], [0, 4]]),
    ],
)
def test_base_shift_register(
    cocotb_test_fixture, data_width, num_points, input_data, expected_output
):
    cocotb_test_fixture.run(
        params=dict(DATA_WIDTH=data_width, NUM_POINTS=num_points),
        defines=dict(),
    )
