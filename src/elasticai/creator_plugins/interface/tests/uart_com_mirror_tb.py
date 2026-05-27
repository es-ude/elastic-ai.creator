import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from pathlib import Path

import elasticai.creator_plugins.interface as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=["uart_com_tb.v", "uart_com.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='UART_COM_MIRROR',
    cocotb_test_module="elasticai.creator_plugins.interface.tests.uart_com_mirror_tb",
    params={'BITRATE': 216, 'NSAMP': 4, 'BITWIDTH': 8},
)


@cocotb.test()
async def uart_com_mirror(dut):
    period_clk = 5
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    baudrate = dut.BITRATE.value.to_unsigned() * dut.NSAMP.value.to_unsigned()

    num_repeats = 4
    data_send_packet = [f'{random.randint(0, 2** bitwidth-1):0{bitwidth}b}' for _ in range(num_repeats)]
    data_get_packet = [f'{0:0{bitwidth}b}' for _ in range(num_repeats)]
    for idx in range(num_repeats-1):
        data_get_packet[idx+1] = data_send_packet[idx]

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.UART_START_FLAG.value = 0
    dut.MASTER_IN.value = 0

    # Start clock and making reset
    cocotb.start_soon(Clock(dut.CLK_SYS, period_clk, unit='ns').start())
    for _ in range(8):
        await RisingEdge(dut.CLK_SYS)
    for idx in range(4):
        await RisingEdge(dut.CLK_SYS)
        dut.RSTN.value = idx % 2
        await RisingEdge(dut.CLK_SYS)
    dut.RSTN.value = 1
    for _ in range(2):
        await RisingEdge(dut.CLK_SYS)

    # make UART transmission
    for data_send, data_check in zip(data_send_packet, data_get_packet):
        # Set Data
        dut.MASTER_IN.value = data_send

        # Set Trigger
        await RisingEdge(dut.CLK_SYS)
        dut.UART_START_FLAG.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.UART_START_FLAG.value = 0

        for _ in range(baudrate):
            await RisingEdge(dut.CLK_SYS)
        assert dut.DRDY.value == 0

        # Check ending
        await RisingEdge(dut.DRDY)
        for _ in range(2*baudrate):
            await RisingEdge(dut.CLK_SYS)
        assert dut.SLAVE_OUT.value == dut.MASTER_IN.value
        assert dut.MASTER_OUT.value == data_check


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
