import random
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from pathlib import Path

import elasticai.creator_plugins.interface as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=["uart_com_fifo_tb.v", "uart_com.v", "uart_fifo.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='UART_COM_FIFO',
    cocotb_test_module="elasticai.creator_plugins.interface.tests.uart_com_fifo_tb",
    params={'BITRATE': 27, 'NSAMP': 4, "FIFO_SIZE": 2, "BITWIDTH": 8},
)


@cocotb.test()
async def uart_com_fifo_external(dut):
    period_clk = 5
    num_bytes = dut.FIFO_SIZE.value.to_unsigned()
    bitwidth = dut.BITWIDTH.value.to_unsigned()
    baudrate = dut.BITRATE.value.to_unsigned() * dut.NSAMP.value.to_unsigned()

    num_repeats = 4
    data_send_packet = [[f'{random.randint(0, 2 ** bitwidth - 1):0{bitwidth}b}' for _ in range(num_bytes)] for _ in range(num_repeats)]
    data_get_packet = [f'{0:0{num_bytes * bitwidth}b}' for _ in range(num_repeats)]
    for idx in range(num_repeats - 1):
        data_get_packet[idx + 1] = "".join(data_send_packet[idx])

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.TX.value = 1
    dut.START_FLAG.value = 0

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

    # make UART package transmission
    for data_send, data_check in zip(data_send_packet, data_get_packet):
        data_get = list()
        for _ in range(baudrate):
            await RisingEdge(dut.CLK_SYS)

        # Sampling TX/RX
        for data_tx in reversed(data_send):
            # Start bit
            dut.TX.value = 0
            for _ in range(baudrate):
                await RisingEdge(dut.CLK_SYS)
            # Data Transmission
            for val in data_tx[::-1]:
                dut.TX.value = val
                for _ in range(int(baudrate/2)):
                    await RisingEdge(dut.CLK_SYS)
                data_get.append(str(dut.RX.value))
                for _ in range(int(baudrate/2)):
                    await RisingEdge(dut.CLK_SYS)
            # Stop bit
            dut.TX.value = 1
            await RisingEdge(dut.mod_drdy)
            for _ in range(int(baudrate/2)):
                await RisingEdge(dut.CLK_SYS)

        for _ in range(int(2*baudrate)):
            await RisingEdge(dut.CLK_SYS)

        assert dut.DRDY.value == 1
        assert dut.FIFO_OUT.value == "".join(data_send[::-1])
        assert "".join(data_check) == "".join(data_get[::-1])

    # Checking Ending
    for _ in range(baudrate):
        await RisingEdge(dut.CLK_SYS)


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
