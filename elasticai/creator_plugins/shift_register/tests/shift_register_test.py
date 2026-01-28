import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray, Range

from elasticai.creator.testing import eai_testbench

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


async def _reset_dut(dut, data_width: int) -> None:
    dut.rst.value = 1
    dut.src_valid.value = 0
    dut.d_in.value = _to_logic_array(0, data_width)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def _write_data(dut, value: int, data_width: int) -> None:
    dut.d_in.value = _to_logic_array(value, data_width)
    dut.src_valid.value = 1
    await RisingEdge(dut.clk)
    dut.src_valid.value = 0
    await RisingEdge(dut.clk)


def _build_expected_shift_register_output(
    input_values: list[int], data_width: int
) -> int:
    max_value = (1 << data_width) - 1
    result = 0
    for value in input_values:
        if value > max_value:
            raise ValueError(
                f"Input value {value} exceeds maximum for {data_width}-bit width (max: {max_value})"
            )
        result = (result << data_width) | value
    return result


async def _assert_register_is_full_and_valid(dut, expected_output: int) -> None:
    actual_output = int(dut.d_out.value)
    assert actual_output == expected_output, (
        f"Output mismatch: got 0x{actual_output:x}, expected 0x{expected_output:x}"
    )
    assert int(dut.valid.value) == 1, "VALID should be high when register is full"


async def _assert_reset_clears_register(dut) -> None:
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)

    assert int(dut.valid.value) == 0, "VALID should be low after reset"
    assert int(dut.d_out.value) == 0, "Output should be zero after reset"


async def _fill_shift_register_with_rising_edges(dut, num_points: int) -> None:
    for _ in range(num_points):
        dut.src_valid.value = 1
        await RisingEdge(dut.clk)
        dut.src_valid.value = 0
        await RisingEdge(dut.clk)


async def _sample_valid_pattern_while_toggling_src_valid(
    dut, num_cycles: int
) -> list[int]:
    pattern = []
    for _ in range(num_cycles):
        dut.src_valid.value = 1
        await RisingEdge(dut.clk)
        pattern.append(int(dut.valid.value))

        dut.src_valid.value = 0
        await RisingEdge(dut.clk)
        pattern.append(int(dut.valid.value))
    return pattern


@ctb.test()
@eai_testbench
async def shift_register_accumulates_data(
    dut,
    data_width: int,
    num_points: int,
    input_data: list[int],
):
    clock = Clock(dut.clk, CLOCK_PERIOD_NS)
    ctb.start_soon(clock.start())
    await _reset_dut(dut, data_width)

    for value in input_data:
        await _write_data(dut, value, data_width)

    expected_output = _build_expected_shift_register_output(input_data, data_width)
    await _assert_register_is_full_and_valid(dut, expected_output)
    await _assert_reset_clears_register(dut)


@ctb.test()
@eai_testbench
async def test_valid_output_follows_src_valid_with_edge_detection(
    dut, data_width: int, num_points: int, input_data: list[int]
):
    clock = Clock(dut.clk, CLOCK_PERIOD_NS)
    ctb.start_soon(clock.start())
    await _reset_dut(dut, data_width)

    await _fill_shift_register_with_rising_edges(dut, num_points)

    num_toggle_cycles = 4
    expected_pattern = [1, 0] * num_toggle_cycles
    actual_pattern = await _sample_valid_pattern_while_toggling_src_valid(
        dut, num_toggle_cycles
    )

    assert actual_pattern == expected_pattern, (
        f"VALID pattern mismatch: got {actual_pattern}, expected {expected_pattern}"
    )


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["data_width", "num_points", "input_data"],
    [
        (4, 4, [1, 2, 3, 4]),
        (3, 3, [2, 5, 7]),
        (5, 2, [12, 23]),
    ],
)
def test_shift_register(
    cocotb_test_fixture,
    data_width,
    num_points,
    input_data,
):
    assert len(input_data) == num_points
    cocotb_test_fixture.write({"input_data": input_data})

    cocotb_test_fixture.run(
        params=dict(DATA_WIDTH=data_width, NUM_POINTS=num_points),
        defines=dict(),
    )
