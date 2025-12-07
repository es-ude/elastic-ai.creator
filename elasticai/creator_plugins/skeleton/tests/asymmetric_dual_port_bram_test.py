import cocotb as ctb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from cocotb.types import LogicArray, Range

from elasticai.creator.file_generation.resource_utils import get_file_from_package
from elasticai.creator.testing import eai_testbench

pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


async def write_to_ram(dut, write_address_width, write_data_width, input_data):
    dut.write_enable.value = 1
    for address, number in enumerate(input_data):
        dut.write_address.value = LogicArray(
            address, Range(write_address_width - 1, "downto", 0)
        )
        dut.d_in.value = LogicArray(number, Range(write_data_width - 1, "downto", 0))
        await RisingEdge(dut.write_clk)
    dut.write_enable.value = 0


def start_clocks(dut):
    read_clk = Clock(dut.read_clk, 10)
    write_clk = Clock(dut.write_clk, 10)
    ctb.start_soon(read_clk.start())
    ctb.start_soon(write_clk.start())


@ctb.test()
@eai_testbench
async def can_read_from_ram(
    dut,
    write_data_width,
    write_address_width,
    write_size,
    read_data_width,
    read_address_width,
    read_size,
    input_data,
    expected,
):
    start_clocks(dut)
    await RisingEdge(dut.write_clk)
    await write_to_ram(dut, write_address_width, write_data_width, input_data)
    result = []
    dut.read_enable.value = 1

    # wait two clock cycles before starting to read
    for address in range(read_size + 2):
        if address < read_size:
            dut.read_address.value = LogicArray(
                address, Range(read_address_width - 1, "downto", 0)
            )
        await RisingEdge(dut.read_clk)
        if address > 1:
            result.append(int(dut.d_out))

    assert result == expected


@ctb.test()
@eai_testbench
async def can_write_to_ram(
    dut,
    write_data_width,
    write_address_width,
    write_size,
    read_data_width,
    read_address_width,
    read_size,
    input_data,
    expected,
):
    start_clocks(dut)
    await RisingEdge(dut.write_clk)
    await write_to_ram(dut, write_address_width, write_data_width, input_data)

    ram_content = [int(x) for x in dut.my_ram.value]
    if min(read_data_width, write_data_width) == write_data_width:
        assert ram_content == input_data
    else:
        assert ram_content == expected


@pytest.mark.simulation
@pytest.mark.parametrize(
    [
        "write_data_width",
        "write_address_width",
        "write_size",
        "read_data_width",
        "read_address_width",
        "read_size",
        "input_data",
        "expected",
    ],
    [
        (2, 2, 4, 4, 1, 2, [1, 2, 3, 0], [9, 3]),
        (4, 1, 2, 2, 2, 4, [9, 3], [1, 2, 3, 0]),
        (
            8,
            3,
            8,
            16,
            2,
            4,
            [0, 1, 1, 1, 1, 8, 15, 15],
            [2**8, 2**8 + 1, 8 * 2**8 + 1, 15 * 2**8 + 15],
        ),
    ],
)
def test_asymmetric_dual_port_bram(
    cocotb_test_fixture,
    write_data_width,
    write_address_width,
    write_size,
    read_data_width,
    read_address_width,
    read_size,
    input_data,
    expected,
):
    with get_file_from_package(
        "elasticai.creator_plugins.skeleton.vhdl", "skeleton_pkg.vhd"
    ) as f:
        cocotb_test_fixture.add_srcs(f)
        cocotb_test_fixture.run(
            params=dict(
                read_address_width=read_address_width,
                read_data_width=read_data_width,
                read_size=read_size,
                write_address_width=write_address_width,
                write_data_width=write_data_width,
                write_size=write_size,
            ),
            defines=dict(),
        )
