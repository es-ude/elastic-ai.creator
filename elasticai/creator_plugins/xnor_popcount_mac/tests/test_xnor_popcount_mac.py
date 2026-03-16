from math import ceil

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from elasticai.creator.testing.cocotb_pytest import (
    CocotbTestFixture,
    eai_testbench,
)
from elasticai.creator.testing.cocotb_stream import (
    ClockReset,
    StreamInterface,
    set_from_bit_string,
)

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


@cocotb.test()
@eai_testbench
async def check_mac(dut, input, weight, expected):
    stream = StreamInterface.from_dut(dut)
    reset = ClockReset.from_dut(dut)
    await _initialize_dut(dut, reset, weight)

    kernel_size, num_channels, num_applications = _derive_stimulus_layout(input, weight)
    input_chunks = _chunk(input, kernel_size)
    expected_chunks = _chunk(expected, num_channels)

    assert len(input_chunks) == num_applications
    assert len(expected_chunks) == num_applications

    collect_task = cocotb.start_soon(
        stream.collect_chunks(expected_count=len(expected_chunks), max_cycles=40)
    )
    await stream.drive_chunks(input_chunks)
    observed_chunks = await collect_task

    dut.src_valid.value = 0
    assert observed_chunks == expected_chunks


async def _initialize_dut(dut, reset: ClockReset, weight: str) -> None:
    cocotb.start_soon(Clock(dut.clk, period=10).start())
    dut.src_valid.value = 0
    dut.dst_ready.value = 0
    set_from_bit_string(dut.weight, weight)
    await reset.reset_active_high()
    dut.en.value = 1
    await RisingEdge(dut.clk)


@pytest.mark.simulation
@pytest.mark.parametrize(
    [
        "input",
        "weight",
        "expected",
    ],
    [
        ("1111", "1111", "1"),
        ("0000", "1111", "0"),
        ("1001", "0101", "1"),
        ("0000", "11110000", "01"),
        ("10011010", "0101", "10"),
        ("1010", "0", "0101"),
    ],
)
def test_mac(cocotb_test_fixture: CocotbTestFixture, input, weight, expected):
    kernel_size, num_channels, _ = _derive_stimulus_layout(input, weight)
    cocotb_test_fixture.run(
        params=dict(
            kernel_size=kernel_size,
            num_channels=num_channels,
        ),
        defines={},
    )


def _derive_stimulus_layout(input: str, weight: str) -> tuple[int, int, int]:
    def div(a: int, b: int) -> int:
        return int(ceil(a / b))

    # we assume a single input channel
    num_channels = div(len(weight), len(input))
    kernel_size = div(len(weight), num_channels)
    num_applications = div(len(input), kernel_size)
    return kernel_size, num_channels, num_applications


def _chunk(input: str, chunk_size: int) -> list[str]:
    if len(input) % chunk_size != 0:
        raise ValueError(
            f"input length {len(input)} is not divisible by chunk_size {chunk_size}"
        )
    return [input[i : i + chunk_size] for i in range(0, len(input), chunk_size)]
