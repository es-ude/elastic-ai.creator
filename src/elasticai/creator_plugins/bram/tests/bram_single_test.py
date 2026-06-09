import random
from pathlib import Path

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

import elasticai.creator_plugins.bram as dut
from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.bram.utils import (
    load_and_plugin,
    translate_path_to_int,
    write_mem_file,
)


@cocotb.test()
@eai_testbench
async def bram_read_write(dut, bitwidth: int, num_params: int, ram: list) -> None:
    arith = int_arithmetic(total_bits=bitwidth, signed=False)
    period_clk = 5

    ram_data_out = [0 for _ in range(num_params)]
    dut.CLK_RAM.value = 0
    dut.EN.value = 0
    dut.WE.value = 0
    dut.ADR.value = 0
    dut.DIN.value = 0

    # Start clock
    cocotb.start_soon(Clock(dut.CLK_RAM, period_clk, unit="ns").start())
    for idx in range(8):
        await RisingEdge(dut.CLK_RAM)
    dut.EN.value = 1

    # --- Part #1: Read only
    dut.WE.value = 0
    await RisingEdge(dut.CLK_RAM)

    ram_data_in = [val.to_unsigned() for val in dut.bram_block.value]
    assert ram_data_in == ram
    for idx in range(num_params):
        await RisingEdge(dut.CLK_RAM)
        dut.ADR.value = idx
        await RisingEdge(dut.CLK_RAM)
        await RisingEdge(dut.CLK_RAM)
        ram_data_out[idx] = dut.DOUT.value.to_unsigned()

    for din, dout in zip(ram, ram_data_out):
        assert din == dout

    # --- Part #2: Write values into RAM
    dut.WE.value = 1
    ram_data_in = [
        random.randint(arith.minimum_as_integer, arith.maximum_as_integer)
        for _ in range(num_params)
    ]
    for idx, val in enumerate(ram_data_in):
        dut.ADR.value = idx
        await RisingEdge(dut.CLK_RAM)
        dut.DIN.value = val
        await RisingEdge(dut.CLK_RAM)

    await RisingEdge(dut.CLK_RAM)
    dut.WE.value = 0
    for _ in range(8):
        await RisingEdge(dut.CLK_RAM)

    # Read values from RAM
    dut.WE.value = 0
    for idx, val in enumerate(ram_data_out):
        await RisingEdge(dut.CLK_RAM)
        dut.ADR.value = idx
        await RisingEdge(dut.CLK_RAM)
        await RisingEdge(dut.CLK_RAM)
        ram_data_out[idx] = dut.DOUT.value
    for din, dout in zip(ram_data_in, ram_data_out):
        assert din == dout


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(12, 32), (6, 16)])
def test_bram_single_port(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int
) -> None:
    temp_path = Path(dut.__file__).parent / "verilog" / "bram_preload.mem"

    cocotb_test_fixture.write({"ram": [val for val in range(num_params)]})
    cocotb_test_fixture.set_top_module_name("BRAM_SINGLE")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "RAMWIDTH": num_params,
            "DATAFILE": translate_path_to_int(temp_path),
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(6, 64), (8, 256)])
def test_bram_single_port_build(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int
) -> None:
    build_path = cocotb_test_fixture.get_artifact_dir() / "verilog"
    temp_path = build_path / "bram_preload.mem"

    write_mem_file(
        path=temp_path,
        data=list(range(num_params)),
        bitwidth=bitwidth,
    )

    load_and_plugin(
        type="bram_single_port",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "RAMWIDTH": num_params,
            "DATAFILE": translate_path_to_int(temp_path),
        },
        packages=["bram"],
        path2save=build_path,
    )

    cocotb_test_fixture.write({"ram": [val for val in range(num_params)]})
    cocotb_test_fixture.set_top_module_name("BRAM_SINGLE_PORT_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
