from cocotb.triggers import RisingEdge


async def _write_current_d_in(dut) -> None:
    dut.src_valid.value = 1
    await RisingEdge(dut.clk)
    dut.src_valid.value = 0


async def _reset_dut(dut) -> None:
    dut.src_valid.value = 0
    dut.dst_ready.value = 0
    dut.rst.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
