from contextlib import ExitStack

import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray, Range

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import eai_testbench

from .helpers import _reset_dut

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"

# Clock period in nanoseconds
CLOCK_PERIOD_NS = 10


async def _write_array(dut, values: list[str], data_width: int) -> None:
    """write to dual shift register by producing valid pulses, just like upstream clients would do."""
    try:
        iter_val = iter(values)
        val = next(iter_val)
        for _ in range(100):
            if dut.ready.value == 1:
                dut.src_valid.value = 1

                dut.d_in.value = LogicArray(
                    val, range=Range(data_width - 1, "downto", 0)
                )
                val = next(iter_val)
            await RisingEdge(dut.clk)
    except StopIteration:
        await RisingEdge(dut.clk)
        dut.src_valid.value = 0
        await RisingEdge(dut.clk)
        return


async def _record_output_for_n_cycles(
    dut, num_points, data_width, n, max_reads
) -> list[list[str]]:
    """read the shift register when valid is high, just like downstream clients would do"""
    values = []

    def get_ouput_as_ints() -> list[str]:
        sliced: list[LogicArray] = [
            dut.d_out.value[(i + 1) * data_width - 1 : i * data_width]
            for i in reversed(range(num_points))
        ]
        return [str(s) for s in sliced]

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
async def dual_shift_register_accumulates_data(
    dut,
    data_width: int,
    num_points_1: int,
    num_points_2: int,
    input_data: list[str],
    sreg2_out: list[list[str]],
):
    clock = Clock(dut.clk, CLOCK_PERIOD_NS)
    ctb.start_soon(clock.start())
    await _reset_dut(dut)

    num_expected_read_cycles = 2 * len(input_data) + 10
    dut.dst_ready.value = 1
    read = ctb.start_soon(
        _record_output_for_n_cycles(
            dut, num_points_2, data_width, num_expected_read_cycles, len(sreg2_out)
        )
    )
    await _write_array(dut, input_data, data_width)
    await read

    actual_output = read.result()
    assert sreg2_out == actual_output


@pytest.mark.simulation
@pytest.mark.parametrize(
    ["data_width", "num_points_1", "num_points_2", "input_data", "sreg2_out"],
    [
        (
            3,
            2,
            2,
            ["000", "001", "010", "100", "001"],
            # sr1 outputs: [["000", "001"], ["001", "010"], ["010", "100"], ["100", "001"]]
            [["001", "011"], ["011", "110"], ["110", "101"]],  # sr2 outputs
        ),
    ],
)
def test_dual_shift_register(
    cocotb_test_fixture,
    data_width,
    num_points_1,
    num_points_2,
    input_data,
    sreg2_out,
):
    def get_file(file):
        return get_file_from_package("elasticai.creator_plugins.shift_register", file)

    with ExitStack() as stack:
        srcs = [
            stack.enter_context(get_file("vhdl/base_shift_register.vhd")),
            stack.enter_context(get_file("tests/dual_shift_register.vhd")),
        ]
        cocotb_test_fixture.set_srcs(srcs)

        cocotb_test_fixture.run(
            params=dict(
                DATA_WIDTH=data_width,
                NUM_POINTS_1=num_points_1,
                NUM_POINTS_2=num_points_2,
            ),
            defines=dict(),
        )
