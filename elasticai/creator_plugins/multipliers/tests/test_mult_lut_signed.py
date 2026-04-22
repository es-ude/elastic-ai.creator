import random
from pathlib import Path

import cocotb
import pytest
from cocotb.triggers import Timer
from hypothesis import HealthCheck, given, settings, strategies

from elasticai.creator.testing import CocotbTestFixture, eai_testbench

from .test_util import build_verilog_design


@cocotb.test()
async def mult_lut_access(dut):
    dut.A.value = 1
    dut.B.value = -2
    await Timer(2, unit="step")
    output = dut.Q.value
    assert output.to_signed() == -2


@cocotb.test()
@eai_testbench
async def mult_lut_random(dut, bitwidth: int, factors: list[int]):
    for a, b in zip(factors[:-1], factors[1:]):
        dut.A.value = a
        dut.B.value = b
        await Timer(2, unit="step")
        output = dut.Q.value
        assert output.to_signed() == a * b


def collect_all_srcs_from_build_dir(build_dir) -> list[Path]:
    all_files = []
    for f in build_dir.iterdir():
        if f.is_file() and f.name.endswith("v"):
            all_files.append(f)
    return all_files


@pytest.mark.simulation
@given(data=strategies.data())
@settings(
    max_examples=3,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=10000,
)
@pytest.mark.parametrize(["bitwidth"], [(3,), (4,), (8,)])
def test_mult_lut_signed(cocotb_test_fixture: CocotbTestFixture, bitwidth: int, data):
    factors = data.draw(
        strategies.lists(
            strategies.integers(
                min_value=-(2 ** (bitwidth - 1)), max_value=(2 ** (bitwidth - 1) - 1)
            ),
            min_size=257,
            max_size=257,
        )
    )
    cocotb_test_fixture.write({"factors": factors})
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
