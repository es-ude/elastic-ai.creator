import random
import cocotb
from cocotb.types import LogicArray
from cocotb.clock import Timer
from pathlib import Path

from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir
import elasticai.creator_plugins.interface as test_dut


cocotb_settings = dict(
    src_files=["spi_slave_woclk.v"],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='SPI_SLAVE_WOCLK',
    cocotb_test_module="elasticai.creator_plugins.interface.tests.spi_slave_woclk_tb",
    params={'BITWIDTH': 6, 'CPOL': 0, 'CPHA': 0, 'MSB': 0}
)


@cocotb.test()
async def spi_slave_access(dut):
    period_clk = 5
    spi_clk_div = 4
    bitwidth = dut.BITWIDTH.value.to_unsigned()

    dut.RSTN.value = 0
    dut.CSN.value = 1
    dut.SCLK.value = dut.CPOL.value.to_unsigned()
    dut.MOSI.value = 0
    dut.DFROM_MIDDLEWARE.value = 0

    # Start clock and making reset
    for _ in range(8):
        await Timer(2*period_clk, unit='ns')
    for idx in range(4):
        await Timer(2*period_clk, unit='ns')
        dut.RSTN.value = idx % 2
        await Timer(2*period_clk, unit='ns')
    dut.RSTN.value = 1
    for _ in range(2):
        await Timer(2*period_clk, unit='ns')
    assert dut.MISO.value == 'z'

    # make SPI transmission
    for _ in range(1):
        data_mosi = f'{random.randint(0, 2** bitwidth-1):0{bitwidth}b}'
        data_miso = list(f'{0:0{bitwidth}b}')
        if not dut.MSB.value:
            data_mosi = data_mosi[::-1]
        dut.DFROM_MIDDLEWARE.value = random.randint(0, 2** bitwidth-1)

        dut.CSN.value = 0
        # Apply data (delay only if CPHA == 1)
        if dut.CPHA.value:
            for _ in range(spi_clk_div):
                await Timer(2*period_clk, unit='ns')
        # Apply data
        for idx, val in enumerate(data_mosi):
            dut.MOSI.value = int(val)

            # Set CLK
            dut.SCLK.value = not dut.CPOL.value.to_unsigned()
            for _ in range(spi_clk_div):
                await Timer(2*period_clk, unit='ns')
            assert dut.DRDY.value == 0

            # Del CLK
            data_miso[idx] = str(dut.MISO.value)
            dut.SCLK.value = dut.CPOL.value.to_unsigned()
            for _ in range(spi_clk_div):
                await Timer(2*period_clk, unit='ns')
        for _ in range(spi_clk_div):
            await Timer(2*period_clk, unit='ns')

        dut.CSN.value = 1
        for _ in range(spi_clk_div):
            await Timer(2*period_clk, unit='ns')

        # Post processing
        assert dut.MISO.value == 'z'
        assert dut.DRDY.value == 1
        assert data_mosi == str(dut.DFOR_MIDDLEWARE.value) if dut.MSB.value else str(dut.DFOR_MIDDLEWARE.value)[::-1]
        print("".join(data_miso), str(dut.DFROM_MIDDLEWARE.value)[::-1], str(dut.DFROM_MIDDLEWARE.value))
        assert "".join(data_miso) == str(dut.DFROM_MIDDLEWARE.value) if dut.MSB.value else str(dut.DFROM_MIDDLEWARE.value)[::-1]

if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
