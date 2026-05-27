import random
import cocotb
from cocotb.types import LogicArray
from cocotb.clock import Clock, Timer
from cocotb.triggers import RisingEdge, FallingEdge
from pathlib import Path

import elasticai.creator_plugins.interface as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=["spi_master.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='SPI_MASTER',
    cocotb_test_module="elasticai.creator_plugins.interface.tests.spi_master_tb",
    params={'BITWIDTH': 12, 'CPOL': 1, 'CPHA': 0, 'MSB': 1, 'SPI_DIV_CLK': 2},
)


@cocotb.test()
async def spi_master_access(dut):
    period_clk = 5
    bitwidth = dut.BITWIDTH.value.to_unsigned()

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
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
    assert dut.MISO.value == 'z'

    # make SPI transmission
    for _ in range(4):
        # Set Data
        data_send = f'{random.randint(0, 2** bitwidth-1):0{bitwidth}b}'
        dut.DATA_TX.value = LogicArray(data_send)
        data_recv = list(f'{random.randint(0, 2** bitwidth-1):0{bitwidth}b}')

        # Set Trigger
        await Timer(period_clk, unit='ns')
        dut.START_FLAG.value = 1
        await Timer(period_clk, unit='ns')
        dut.START_FLAG.value = 0
        await Timer(period_clk, unit='ns')

        # Sampling MOSI
        data_get = list(f'{0:0{bitwidth}b}')
        for idx in range(bitwidth):
            if dut.CPHA.value:
                if dut.CPOL.value:
                    await FallingEdge(dut.SCLK)
                else:
                    await RisingEdge(dut.SCLK)
                dut.MISO.value = int(data_recv[idx])
                if dut.CPOL.value:
                    await RisingEdge(dut.SCLK)
                else:
                    await FallingEdge(dut.SCLK)
                data_get[idx] = str(dut.MOSI.value)
            else:
                dut.MISO.value = int(data_recv[idx])
                if dut.CPOL.value:
                    await FallingEdge(dut.SCLK)
                else:
                    await RisingEdge(dut.SCLK)
                data_get[idx] = str(dut.MOSI.value)
                if dut.CPOL.value:
                    await RisingEdge(dut.SCLK)
                else:
                    await FallingEdge(dut.SCLK)
            assert dut.CSN.value == 0

        # Checking Ending
        await RisingEdge(dut.DRDY)
        await RisingEdge(dut.CLK_SYS)
        assert dut.DRDY.value == 1
        assert dut.CSN.value == 1
        data_chck_send = "".join(data_get) if dut.MSB.value else "".join(data_get[::-1])
        data_chck_recv = "".join(data_recv) if dut.MSB.value else "".join(data_recv[::-1])
        print(f"Send {data_chck_send} (should be {data_send})")
        print(f"Receive {dut.DATA_RX.value} (should be {data_chck_recv})")
        assert data_send == data_chck_send
        assert dut.DATA_RX.value == data_chck_recv


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
