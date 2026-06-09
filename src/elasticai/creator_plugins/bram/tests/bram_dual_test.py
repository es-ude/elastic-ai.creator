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
    dut.WE_A.value = 0
    dut.WE_B.value = 0
    dut.ADR_A.value = 0
    dut.ADR_B.value = 0
    dut.DIN_A.value = 0
    dut.DIN_B.value = 0

    # Start clock
    cocotb.start_soon(Clock(dut.CLK_RAM, period_clk, unit="ns").start())
    for idx in range(8):
        await RisingEdge(dut.CLK_RAM)
    dut.EN.value = 1

    # --- Part #1: Read only (Port A)
    dut.WE_A.value = 0
    dut.WE_B.value = 0
    await RisingEdge(dut.CLK_RAM)

    ram_data_in = [val.to_unsigned() for val in dut.bram_block.value]
    assert ram_data_in == ram
    for idx in range(num_params // 2):
        await RisingEdge(dut.CLK_RAM)
        dut.ADR_A.value = 2 * idx
        dut.ADR_B.value = 2 * idx + 1
        await RisingEdge(dut.CLK_RAM)
        await RisingEdge(dut.CLK_RAM)
        ram_data_out[2 * idx] = dut.DOUT_A.value.to_unsigned()
        ram_data_out[2 * idx + 1] = dut.DOUT_B.value.to_unsigned()

    for din, dout in zip(ram, ram_data_out):
        assert din == dout

    # --- Part #2: Write values into RAM (Port A) and Read Port B
    dut.WE_A.value = 1
    dut.WE_B.value = 0
    ram_data_in = [
        random.randint(arith.minimum_as_integer, arith.maximum_as_integer)
        for _ in range(num_params)
    ]
    for idx, val in enumerate(ram_data_in):
        dut.ADR_A.value = idx
        await RisingEdge(dut.CLK_RAM)
        dut.DIN_A.value = val
        await RisingEdge(dut.CLK_RAM)

    await RisingEdge(dut.CLK_RAM)
    dut.WE_A.value = 0
    dut.WE_B.value = 0
    for _ in range(8):
        await RisingEdge(dut.CLK_RAM)

    # Read values from RAM
    dut.WE_A.value = 0
    dut.WE_B.value = 0
    for idx, val in enumerate(ram_data_out):
        await RisingEdge(dut.CLK_RAM)
        dut.ADR_B.value = idx
        await RisingEdge(dut.CLK_RAM)
        await RisingEdge(dut.CLK_RAM)
        ram_data_out[idx] = dut.DOUT_B.value
    for din, dout in zip(ram_data_in, ram_data_out):
        assert din == dout

    # --- Part #3: Write values into RAM (Port B) and Read Port A
    dut.WE_A.value = 0
    dut.WE_B.value = 1
    ram_data_in = [
        random.randint(arith.minimum_as_integer, arith.maximum_as_integer)
        for _ in range(num_params)
    ]
    for idx, val in enumerate(ram_data_in):
        dut.ADR_B.value = idx
        await RisingEdge(dut.CLK_RAM)
        dut.DIN_B.value = val
        await RisingEdge(dut.CLK_RAM)

    await RisingEdge(dut.CLK_RAM)
    dut.WE_A.value = 0
    dut.WE_B.value = 0
    for _ in range(8):
        await RisingEdge(dut.CLK_RAM)

    # Read values from RAM
    dut.WE_A.value = 0
    dut.WE_B.value = 0
    for idx, val in enumerate(ram_data_out):
        await RisingEdge(dut.CLK_RAM)
        dut.ADR_A.value = idx
        await RisingEdge(dut.CLK_RAM)
        await RisingEdge(dut.CLK_RAM)
        ram_data_out[idx] = dut.DOUT_A.value
    for din, dout in zip(ram_data_in, ram_data_out):
        assert din == dout


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(12, 32), (6, 16)])
def test_bram_dual_port(
    cocotb_test_fixture: CocotbTestFixture, bitwidth: int, num_params: int
) -> None:
    temp_path = Path(dut.__file__).parent / "verilog" / "bram_preload.mem"

    cocotb_test_fixture.write({"ram": [val for val in range(num_params)]})
    cocotb_test_fixture.set_top_module_name("BRAM_DUAL")
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
def test_bram_dual_port_build(
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
        type="bram_dual_port",
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
    cocotb_test_fixture.set_top_module_name("BRAM_DUAL_PORT_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
