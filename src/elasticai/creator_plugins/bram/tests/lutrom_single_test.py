from random import randint

import cocotb
import pytest
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

from elasticai.creator.arithmetic import int_arithmetic, int_converter
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.bram.utils import load_and_plugin


@cocotb.test()
@eai_testbench
async def lutrom_read(
    dut, bitwidth: int, num_params: int, is_signed: bool, ram: list
) -> None:
    period_clk = 5

    ram_data_out = [0 for _ in range(num_params)]
    dut.CLK_SYS.value = 0
    dut.EN.value = 0
    dut.ADR.value = 0

    # Start clock
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit="ns").start())
    for idx in range(8):
        await RisingEdge(dut.CLK_SYS)
    dut.EN.value = 1

    # --- Part #1: Read only
    await RisingEdge(dut.CLK_SYS)

    for idx in range(num_params):
        await RisingEdge(dut.CLK_SYS)
        dut.ADR.value = idx
        await RisingEdge(dut.CLK_SYS)
        await RisingEdge(dut.CLK_SYS)
        ram_data_out[idx] = dut.DOUT.value.to_unsigned()

    print(ram_data_out)
    print(ram)
    for din, dout in zip(ram, ram_data_out):
        assert din == dout


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(8, 8)])
@pytest.mark.parametrize("is_signed", [False])
def test_lutrom_single_port(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    is_signed: bool,
) -> None:
    cocotb_test_fixture.write({"ram": [val for val in range(num_params)]})
    cocotb_test_fixture.set_top_module_name("LUTROM")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "ROMWIDTH": num_params,
        },
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth, num_params", [(6, 64), (8, 256)])
@pytest.mark.parametrize("is_signed", [False])
def test_lutrom_single_port_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_params: int,
    is_signed: bool,
) -> None:
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    conv = int_converter(total_bits=bitwidth, signed=is_signed)
    data = [
        randint(arith.minimum_as_integer, arith.maximum_as_integer)
        for _ in range(num_params)
    ]
    data0 = data.copy()
    data0.reverse()

    build_path = cocotb_test_fixture.get_artifact_dir() / "verilog"

    load_and_plugin(
        type="lutrom_single_port",
        id="0",
        params={
            "BITWIDTH": bitwidth,
            "ROMWIDTH": num_params,
            "LUTROM_DATA": conv.integer_to_binary_string_array_verilog(data0),
        },
        packages=["bram"],
        path2save=build_path,
    )

    cocotb_test_fixture.write({"ram": data})
    cocotb_test_fixture.set_top_module_name("LUTROM_SINGLE_PORT_0")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_artifact_dir("verilog/*.v")
    cocotb_test_fixture.run(params={}, defines={})
