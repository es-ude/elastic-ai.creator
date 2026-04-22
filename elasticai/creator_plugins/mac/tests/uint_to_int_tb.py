import random
import cocotb
from cocotb.triggers import Timer
from pathlib import Path

import elasticai.creator_plugins.mac as test_dut
from elasticai.creator.testing.cocotb_runner import run_cocotb_sim_for_src_dir 


cocotb_settings = dict(
    src_files=['uint_to_int.v'],
    path2src=Path(test_dut.__file__).parent / 'verilog',
    top_module_name='UNSIGNED_TO_SIGNED',
    cocotb_test_module="elasticai.creator_plugins.mac.tests.uint_to_int_tb",
    params={'BITWIDTH': 4}
)


@cocotb.test()
async def int_transform_access(dut):
    dut.A.value = 1
    await Timer(2, unit='step')
    assert dut.Q.value.to_signed() == 1


@cocotb.test()
async def int_transform_random_positive(dut):
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned()-1)
    for _ in range(100):
        val = random.randint(0, valrange - 1)
        dut.A.value = val
        await Timer(2, unit='step')
        assert dut.Q.value.to_signed() == val


@cocotb.test()
async def int_transform_random_negative(dut):
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned()-1)
    for _ in range(100):
        val = random.randint(valrange, 2*valrange-1)
        dut.A.value = val
        await Timer(2, unit='step')
        assert dut.Q.value.to_signed() == val - 2*valrange


if __name__ == "__main__":
    run_cocotb_sim_for_src_dir (**cocotb_settings)
