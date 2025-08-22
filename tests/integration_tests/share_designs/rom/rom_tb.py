import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from elasticai.creator.testing.cocotb_prepare import read_testdata


@cocotb.test()
async def rom_call_test(dut):
    data = read_testdata(dut._name)
    clock_period_ns = 10

    dut.en.value = 0
    dut.clk.value = 0
    dut.addr.value = 0

    cocotb.start_soon(Clock(dut.clk, period=clock_period_ns, units="ns").start())
    await Timer(4 * clock_period_ns, units="ns")
    await RisingEdge(dut.clk)
    dut.en.value = 1

    # --- Getting data
    for addr, data_chck in enumerate(data["data"]):
        dut.addr.value = addr
        await RisingEdge(dut.clk)
        assert dut.ROM.value[addr].signed_integer == data_chck
        await RisingEdge(dut.clk)

    # for addr in range(len(data["data"]), 2**):


@cocotb.test()
async def rom_content_read(dut):
    data = read_testdata(dut._name)
    dut.en.value = 0
    dut.clk.value = 0
    dut.addr.value = 0

    for rom_b, json_b in zip(dut.ROM.value, data["data"]):
        assert rom_b.signed_integer == json_b
