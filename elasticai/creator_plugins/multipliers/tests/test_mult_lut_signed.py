import random
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.testing.cocotb_pytest import CocotbTestFixture

from .test_util import build_verilog_design

# cocotb_settings = dict(
#     src_files=["mult_lut_signed.v", "adder_full.v", "adder_half.v"],
#     path2src=Path(test_dut.__file__).parent / "verilog",
#     top_module_name="MULT_LUT_SIGNED",
#     cocotb_test_module="elasticai.creator_plugins.mult.tests.mult_lut_signed_tb",
#     params={"BITWIDTH": 8},
#     en_debug_mode=True,
# )
#
pytest_plugins = "elasticai.creator.testing.cocotb_pytest"


@cocotb.test()
async def mult_lut_access(dut):
    dut.A.value = 1
    dut.B.value = -2
    await Timer(2, unit="step")
    output = dut.Q.value
    assert output.to_signed() == -2


@cocotb.test()
async def mult_lut_random(dut):
    valrange = 2 ** (dut.BITWIDTH.value.to_unsigned() - 1)
    for _ in range(256):
        A = random.randint(-valrange, valrange - 1)
        B = random.randint(-valrange, valrange - 1)
        dut.A.value = A
        dut.B.value = B
        await Timer(2, unit="step")
        output = dut.Q.value
        assert output.to_signed() == A * B


def collect_all_srcs_from_build_dir(build_dir) -> list[Path]:
    all_files = []
    for f in build_dir.iterdir():
        if f.is_file() and f.name.endswith("v"):
            all_files.append(f)
    return all_files


@pytest.mark.simulation
@pytest.mark.parametrize(["bitwidth"], [(3,)])
def test_mult_lut_signed(cocotb_test_fixture: CocotbTestFixture, bitwidth):

    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"
    build_dir.mkdir(exist_ok=True)
    build_verilog_design(
        type="mult_lut_signed",
        id=f"{bitwidth}",
        params={"BITWIDTH": bitwidth},
        packages=["multipliers"],
        path2save=build_dir,
    )
    cocotb_test_fixture.set_top_module_name(f"MULT_LUT_SIGNED_{bitwidth}")

    cocotb_test_fixture.set_srcs(collect_all_srcs_from_build_dir(build_dir))
    cocotb_test_fixture.run(params={}, defines={})
