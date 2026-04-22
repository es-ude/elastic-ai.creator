from contextlib import ExitStack

import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import eai_testbench

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


def _bytes_per_word(data_width: int) -> int:
    return max(1, (data_width + 7) // 8)


def _pack_words_to_bytes_le(words: list[int], data_width: int) -> list[int]:
    mask = (1 << data_width) - 1
    transport_width = _bytes_per_word(data_width) * 8
    value = 0
    for i, word in enumerate(words):
        value |= (int(word) & mask) << (i * transport_width)

    num_bytes = len(words) * _bytes_per_word(data_width)
    return list(value.to_bytes(num_bytes, byteorder="little", signed=False))


async def _write_input_bytes(dut, payload: list[int]) -> None:
    dut.input_wr_enable.value = 0
    dut.input_wr_address.value = 0
    dut.input_wr_data.value = 0

    for addr, byte in enumerate(payload):
        dut.input_wr_address.value = addr
        dut.input_wr_data.value = byte
        dut.input_wr_enable.value = 1
        await RisingEdge(dut.clk)

    dut.input_wr_enable.value = 0


@ctb.test()
@eai_testbench
async def streams_full_frame_words(dut, data_width, data_depth, payload_words):
    ctb.start_soon(Clock(dut.clk, 10).start())

    dut.rst.value = 1
    dut.start.value = 0
    dut.input_wr_enable.value = 0
    dut.input_wr_address.value = 0
    dut.input_wr_data.value = 0

    await RisingEdge(dut.clk)
    dut.rst.value = 0

    payload_bytes = _pack_words_to_bytes_le(payload_words, data_width)
    await _write_input_bytes(dut, payload_bytes)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    received: list[int] = []
    done_seen = False
    for _ in range(128):
        await RisingEdge(dut.clk)
        if int(dut.stream_valid.value) == 1:
            received.append(int(dut.stream_data.value))
        if int(dut.stream_done.value) == 1:
            done_seen = True
            break

    assert done_seen
    assert received == [w & ((1 << data_width) - 1) for w in payload_words]


@ctb.test()
@eai_testbench
async def reset_aborts_streaming(dut, data_width, data_depth, payload_words):
    ctb.start_soon(Clock(dut.clk, 10).start())

    dut.rst.value = 1
    dut.start.value = 0
    dut.input_wr_enable.value = 0
    dut.input_wr_address.value = 0
    dut.input_wr_data.value = 0

    await RisingEdge(dut.clk)
    dut.rst.value = 0

    payload_bytes = _pack_words_to_bytes_le(payload_words, data_width)
    await _write_input_bytes(dut, payload_bytes)

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    dut.rst.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    done_seen = False
    for _ in range(64):
        await RisingEdge(dut.clk)
        if int(dut.stream_done.value) == 1:
            done_seen = True
            break

    assert not done_seen


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("data_width", "payload_words"),
    (
        (5, [0x01, 0x1F, 0x12, 0x08]),
        (8, [0x01, 0x02, 0x03, 0x04]),
        (11, [0x001, 0x7FF, 0x345, 0x2AA]),
    ),
)
def test_skeleton_input_adapter(cocotb_test_fixture, data_width, payload_words):
    data_depth = 4

    with ExitStack() as stack:
        files = [
            "skeleton_pkg.vhd",
            "asymmetric_dual_port_bram.vhd",
            "buffered_network_input_reader.vhd",
            "skeleton_input_adapter.vhd",
        ]
        srcs = []
        for filename in files:
            srcs.append(
                stack.enter_context(
                    get_file_from_package(
                        "elasticai.creator_plugins.skeleton.vhdl", filename
                    )
                )
            )

        cocotb_test_fixture.set_top_module_name("skeleton_input_adapter")
        cocotb_test_fixture.set_srcs(tuple(srcs))
        cocotb_test_fixture.write(
            {
                "data_width": data_width,
                "data_depth": data_depth,
                "payload_words": payload_words,
            }
        )
        cocotb_test_fixture.run(
            params=dict(
                data_in_width=data_width,
                data_in_depth=data_depth,
            ),
            defines=dict(),
        )
