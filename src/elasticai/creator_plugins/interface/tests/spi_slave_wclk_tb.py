import random
import cocotb
from cocotb.types import LogicArray
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
from pathlib import Path
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.interface as test_dut


cocotb_settings = dict(
    src_files=["spi_slave_wclk.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='SPI_SLAVE_WCLK',
    cocotb_test_module="elasticai.creator_plugins.interface.tests.spi_slave_wclk_tb",
    params={'BITWIDTH': 6, 'CPOL': 0, 'CPHA': 0, 'MSB': 1}
)


@cocotb.test()
async def spi_slave_access(dut):
    period_clk = 5
    spi_clk_div = 4

    dut.CLK_SYS.value = 0
    dut.RSTN.value = 0
    dut.CSN.value = 1
    dut.SCLK.value = dut.CPOL.value.to_unsigned()
    dut.MOSI.value = 0
    dut.DFROM_MIDDLEWARE.value = random.randint(0, 2** dut.BITWIDTH.value.to_unsigned()-1)

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
    for _ in range(1):
        data_send = f'{random.randint(0, 2** dut.BITWIDTH.value.to_unsigned()-1):0{dut.BITWIDTH.value.to_unsigned()}b}'
        if not dut.MSB.value:
            data_send = data_send[::-1]
        data_get = list(f'{0:0{dut.BITWIDTH.value.to_unsigned()}b}')

        dut.CSN.value = 0
        # Apply data (delay only if CPHA == 1)
        if dut.CPHA.value:
            for _ in range(spi_clk_div):
                await RisingEdge(dut.CLK_SYS)
        # Apply data
        for idx, val in enumerate(data_send):
            dut.MOSI.value = int(val)

            # Set CLK
            dut.SCLK.value = not dut.CPOL.value
            for _ in range(spi_clk_div):
                await RisingEdge(dut.CLK_SYS)

            # Del CLK
            dut.SCLK.value = dut.CPOL.value.to_unsigned()
            data_get[idx] = str(dut.MISO.value)
            for _ in range(spi_clk_div):
                await RisingEdge(dut.CLK_SYS)

        for _ in range(spi_clk_div):
            await RisingEdge(dut.CLK_SYS)
        dut.CSN.value = 1
        for _ in range(spi_clk_div):
            await RisingEdge(dut.CLK_SYS)

        assert dut.MISO.value == 'z'
        assert dut.DRDY.value == 1
        assert data_send == dut.DFOR_MIDDLEWARE.value if dut.MSB.value else LogicArray(str(dut.DFOR_MIDDLEWARE.value)[::-1]).to_unsigned()
        assert "".join(data_get) == dut.DFROM_MIDDLEWARE.value if dut.MSB.value else LogicArray(str(dut.DFOR_MIDDLEWARE.value)[::-1]).to_unsigned()


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
