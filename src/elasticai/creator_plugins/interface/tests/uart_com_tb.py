import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from pathlib import Path

import elasticai.creator_plugins.interface as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=["uart_com.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='UART_COM',
    cocotb_test_module="elasticai.creator_plugins.interface.tests.uart_com_tb",
    params={'BITRATE': 216, 'NSAMP': 4, 'BITWIDTH': 5},
)


@cocotb.test()
async def uart_com(dut):
    period_clk = 5
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    baudrate = dut.BITRATE.value.to_unsigned() * dut.NSAMP.value.to_unsigned()

    num_repeats = 4
    data_send_packet = [f'{random.randint(0, 2 ** bitwidth - 1):0{bitwidth}b}' for _ in range(num_repeats)]
    data_get_packet = [f'{0:0{bitwidth}b}' for _ in range(num_repeats)]
    for idx in range(num_repeats - 1):
        data_get_packet[idx + 1] = data_send_packet[idx]

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.UART_START_FLAG.value = 0
    dut.RX.value = 1
    dut.UART_DIN.value = 0

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
        dut.UART_DIN.value = dut.UART_DOUT.value

        # Set Trigger
        await RisingEdge(dut.CLK_SYS)
        dut.UART_START_FLAG.value = 1
        await RisingEdge(dut.CLK_SYS)
        dut.UART_START_FLAG.value = 0

        await RisingEdge(dut.CLK_SYS)
        assert dut.UART_RDY.value == 0

        # Sampling TX/RX
        data_get = list()
        # Start bit
        dut.RX.value = 0
        for _ in range(baudrate):
            await RisingEdge(dut.CLK_SYS)
        # Data Transmission
        for val in data_send[::-1]:
            dut.RX.value = val
            for _ in range(int(baudrate/2)):
                await RisingEdge(dut.CLK_SYS)
            data_get.append(str(dut.TX.value))
            for _ in range(int(baudrate/2)):
                await RisingEdge(dut.CLK_SYS)
        # Stop bit
        dut.RX.value = 1

        # Checking Ending
        await RisingEdge(dut.UART_RDY)
        for _ in range(baudrate):
            await RisingEdge(dut.CLK_SYS)
        assert dut.UART_RDY.value == 1

        assert dut.UART_DOUT.value == "".join(data_send)
        assert data_check == "".join(data_get[::-1])


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
