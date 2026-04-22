from contextlib import ExitStack
from pathlib import Path

import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import eai_testbench

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


def _write_delayed_identity_network(path: Path, data_width: int) -> None:
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
                "  signal d_reg : std_logic_vector("
                + str(data_width - 1)
                + " downto 0) := (others => '0');",
                "  signal v_reg : std_logic := '0';",
                "begin",
                "  process(clk)",
                "  begin",
                "    if rising_edge(clk) then",
                "      if rst = '1' then",
                "        d_reg <= (others => '0');",
                "        v_reg <= '0';",
                "      else",
                "        d_reg <= D_IN;",
                "        v_reg <= SRC_VALID and DST_READY and EN;",
                "      end if;",
                "    end if;",
                "  end process;",
                "  D_OUT <= d_reg;",
                "  VALID <= v_reg;",
                "  READY <= DST_READY;",
                "end architecture;",
                "",
            )
        ),
        encoding="utf-8",
    )


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


async def _read_output_bytes(dut, num_bytes: int) -> list[int]:
    result: list[int] = []
    dut.output_rd_enable.value = 0
    dut.output_rd_address.value = 0

    for addr in range(num_bytes):
        dut.output_rd_address.value = addr
        dut.output_rd_enable.value = 1
        await RisingEdge(dut.clk)

        dut.output_rd_enable.value = 0
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        result.append(int(dut.output_rd_data.value))

    return result


@ctb.test()
@eai_testbench
async def handles_non_simultaneous_done_pulses(
    dut, data_width, data_depth, payload_words
):
    ctb.start_soon(Clock(dut.clk, 10).start())

    dut.rst.value = 1
    dut.network_enable.value = 0
    dut.input_wr_enable.value = 0
    dut.input_wr_address.value = 0
    dut.input_wr_data.value = 0
    dut.output_rd_enable.value = 0
    dut.output_rd_address.value = 0

    await RisingEdge(dut.clk)
    dut.rst.value = 0

    payload_bytes = _pack_words_to_bytes_le(payload_words, data_width)
    await _write_input_bytes(dut, payload_bytes)

    dut.network_enable.value = 1
    await RisingEdge(dut.clk)
    dut.network_enable.value = 0

    done_seen = False
    for _ in range(1024):
        await RisingEdge(dut.clk)
        if int(dut.done.value) == 1:
            done_seen = True
            break

    assert done_seen

    readback = await _read_output_bytes(dut, len(payload_bytes))
    assert readback == payload_bytes


@pytest.mark.simulation
@pytest.mark.parametrize(
    ("data_width", "payload_words"),
    (
        (5, [0x01, 0x1F, 0x12, 0x08]),
        (8, [0x01, 0x02, 0x03, 0x04]),
        (11, [0x001, 0x7FF, 0x345, 0x2AA]),
    ),
)
def test_skeleton_inference_controller(cocotb_test_fixture, data_width, payload_words):
    data_depth = 4

    network_file = cocotb_test_fixture.get_artifact_dir() / "network.vhd"
    _write_delayed_identity_network(network_file, data_width)

    with ExitStack() as stack:
        pkg = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl", "skeleton_pkg.vhd"
            )
        )
        bram = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl",
                "asymmetric_dual_port_bram.vhd",
            )
        )
        input_reader = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl",
                "buffered_network_input_reader.vhd",
            )
        )
        frame_ingress = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl",
                "skeleton_input_adapter.vhd",
            )
        )
        network_runner = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl",
                "skeleton_network_runner.vhd",
            )
        )
        inference_controller = stack.enter_context(
            get_file_from_package(
                "elasticai.creator_plugins.skeleton.vhdl",
                "skeleton_inference_controller.vhd",
            )
        )

        cocotb_test_fixture.set_top_module_name("skeleton_inference_controller")
        cocotb_test_fixture.set_srcs(
            (
                pkg,
                bram,
                input_reader,
                frame_ingress,
                network_runner,
                inference_controller,
                network_file,
            )
        )
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
                data_out_width=data_width,
                data_out_depth=data_depth,
            ),
            defines=dict(),
        )
