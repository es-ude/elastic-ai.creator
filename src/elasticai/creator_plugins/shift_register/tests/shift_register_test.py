import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray, Range

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import eai_testbench

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


# Clock period in nanoseconds
CLOCK_PERIOD_NS = 10


async def _reset_dut(dut) -> None:
    dut.src_valid.value = 0
    dut.dst_ready.value = 0
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


def _to_logic_array(value: int, width: int) -> LogicArray:
    max_value = (1 << width) - 1
    if value > max_value:
        raise ValueError(
            f"Value {value} exceeds maximum for {width}-bit width (max: {max_value})"
        )
    return LogicArray(value, Range(width - 1, "downto", 0))


async def _write_array(dut, values: list[int], data_width: int) -> None:
    """write to dual shift register by producing valid pulses, just like upstream clients would do."""
    try:
        iter_val = iter(values)
        val = next(iter_val)
        for _ in range(100):
            if dut.ready.value == 1:
                dut.src_valid.value = 1

                dut.d_in.value = _to_logic_array(val, data_width)
                val = next(iter_val)
            await RisingEdge(dut.clk)
    except StopIteration:
        await RisingEdge(dut.clk)
        dut.src_valid.value = 0
        await RisingEdge(dut.clk)
        return


async def _record_output_for_n_cycles(
    dut, num_points, data_width, n, max_reads
) -> list[list[int]]:
    """read the shift register when valid is high, just like downstream clients would do"""
    values = []

    def get_ouput_as_ints() -> list[int]:
        sliced: list[LogicArray] = [
            dut.d_out.value[(i + 1) * data_width - 1 : i * data_width]
            for i in reversed(range(num_points))
        ]
        return [s.to_unsigned() for s in sliced]

    dut.dst_ready.value = 1
    for _ in range(n):
        await RisingEdge(dut.clk)
        # Capture output on rising edge of valid (0->1 transition)
        if dut.valid.value == 1:
            values.append(get_ouput_as_ints())
        if len(values) == max_reads:
            dut.dst_ready.value = 0
            await RisingEdge(dut.clk)
            break

    return values


@ctb.test()
@eai_testbench
async def shift_register_accumulates_data(
    dut,
    data_width: int,
    num_points: int,
    skip: int,
    input_data: list[int],
    expected_output: list[list[int]],
):
    clock = Clock(dut.clk, CLOCK_PERIOD_NS)
    ctb.start_soon(clock.start())
    await _reset_dut(dut)

    # we need two cycles to read each data point and one cycle for the valid signal and data to propagate
    num_expected_read_cycles = 2 * len(input_data) * skip + 1
    dut.en.value = 1
    read = ctb.start_soon(
        _record_output_for_n_cycles(
            dut,
            num_points,
            data_width,
            num_expected_read_cycles,
            max_reads=len(expected_output),
        )
    )
    await _write_array(dut, input_data, data_width)
    await read

    actual_output = read.result()
    assert expected_output == actual_output


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["data_width", "num_points", "skip", "input_data", "expected_output"],
    [
        (4, 4, 1, [1, 2, 3, 4, 5], [[1, 2, 3, 4], [2, 3, 4, 5]]),
        (4, 4, 2, [1, 2, 3, 4, 5, 6, 7], [[1, 3, 5, 7]]),
        (3, 3, 3, [2, 5, 7, 3, 1, 0, 6], [[2, 3, 6]]),
        (5, 2, 4, [12, 23, 0, 13, 14, 15, 18, 17, 19], [[12, 14], [14, 19]]),
    ],
)
def test_shift_register(
    cocotb_test_fixture, data_width, num_points, skip, input_data, expected_output
):
    with get_file_from_package(
        "elasticai.creator_plugins.shift_register", "vhdl/base_shift_register.vhd"
    ) as f:
        cocotb_test_fixture.add_srcs(f)
        cocotb_test_fixture.run(
            params=dict(DATA_WIDTH=data_width, NUM_POINTS=num_points, SKIP=skip),
            defines=dict(),
        )
