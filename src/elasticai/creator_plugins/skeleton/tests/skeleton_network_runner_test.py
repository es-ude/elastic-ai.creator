from contextlib import ExitStack
from pathlib import Path

import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import eai_testbench

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


def _write_identity_network(path: Path, data_width: int) -> None:
    path.write_text(
        "\n".join(
            (
                "library ieee;",
                "use ieee.std_logic_1164.all;",
                "",
                "entity network is",
                "  port (",
                "    CLK : in std_logic;",
                f"    D_IN : in std_logic_vector({data_width - 1} downto 0);",
                f"    D_OUT : out std_logic_vector({data_width - 1} downto 0);",
                "    SRC_VALID : in std_logic;",
                "    RST : in std_logic;",
                "    VALID : out std_logic;",
                "    READY : out std_logic;",
                "    DST_READY : in std_logic;",
                "    EN : in std_logic",
                "  );",
                "end entity;",
                "",
                "architecture rtl of network is",
                "begin",
                "  D_OUT <= D_IN;",
                "  VALID <= SRC_VALID and DST_READY and EN;",
                "  READY <= DST_READY;",
                "end architecture;",
                "",
            )
        ),
        encoding="utf-8",
    )


@ctb.test()
@eai_testbench
async def emits_output_words_and_done(dut, data_width, data_depth, payload_words):
    ctb.start_soon(Clock(dut.clk, 10).start())

    dut.rst.value = 1
    dut.start.value = 0
    dut.stream_valid.value = 0
    dut.stream_data.value = 0

    await RisingEdge(dut.clk)
    dut.rst.value = 0

    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    payload_iter = iter(payload_words)
    outputs: list[int] = []
    done_seen = False

    for _ in range(128):
        await RisingEdge(dut.clk)

        try:
            value = next(payload_iter)
            dut.stream_data.value = value
            dut.stream_valid.value = 1
        except StopIteration:
            dut.stream_valid.value = 0

        if int(dut.output_word_valid.value) == 1:
            outputs.append(int(dut.output_word_data.value) & ((1 << data_width) - 1))

        if int(dut.done.value) == 1:
            done_seen = True
            break

    assert done_seen
    assert outputs == payload_words


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("data_width", "payload_words"),
    (
        (5, [0x01, 0x1F, 0x12, 0x08]),
        (8, [0x01, 0x02, 0x03, 0x04]),
        (11, [0x001, 0x7FF, 0x345, 0x2AA]),
    ),
)
def test_skeleton_network_runner(cocotb_test_fixture, data_width, payload_words):
    data_depth = 4

    network_file = cocotb_test_fixture.get_artifact_dir() / "network.vhd"
    _write_identity_network(network_file, data_width)

    with ExitStack() as stack:
        pkg = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl", "skeleton_pkg.vhd"
            )
        )
        runner = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl",
                "skeleton_network_runner.vhd",
            )
        )

        cocotb_test_fixture.set_top_module_name("skeleton_network_runner")
        cocotb_test_fixture.set_srcs((pkg, runner, network_file))
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
                data_out_width=data_width,
                data_out_depth=data_depth,
            ),
            defines=dict(),
        )
