import cocotb
import pytest
from cocotb.triggers import Timer

from elasticai.creator.arithmetic import int_arithmetic
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins.adders.utils import (
    collect_all_srcs_from_build_dir,
    load_and_plugin,
)


def dtct_model(
    a: int, bitwidth: int, num_adders: int, signed: bool
) -> tuple[int, int, int]:
    valrange = (
        [0, 2 ** (bitwidth - num_adders) - 1]
        if not signed
        else [-(2 ** (bitwidth - 1 - num_adders)), 2 ** (bitwidth - 1 - num_adders) - 1]
    )
    down = a < valrange[0]
    upper = a > valrange[1]
    return 1 if down else 0, 1 if upper else 0, 0 if down or upper else 1


@cocotb.test()
@eai_testbench
async def violation_chck(dut, bitwidth: int, num_adders: int, is_signed: bool):
    arith = int_arithmetic(total_bits=bitwidth, signed=is_signed)
    valrange = [arith.minimum_as_integer, arith.maximum_as_integer]
    if not is_signed:
        test = valrange.copy()
        test.extend(
            [
                idx
                for idx in range(
                    2 ** (bitwidth - num_adders) - 2**num_adders,
                    2 ** (bitwidth - num_adders) + 2**num_adders,
                )
            ]
        )
    else:
        test = valrange.copy()
        test.extend(
            [
                idx
                for idx in range(
                    -(2 ** (bitwidth - num_adders)) + 2 * num_adders, -(2**num_adders)
                )
            ]
        )
        test.extend(
            [
                idx
                for idx in range(
                    2**num_adders, +(2 ** (bitwidth - num_adders)) - 2**num_adders
                )
            ]
        )

    for val in test:
        dut.A.value = (
            arith.clamp(val) if not is_signed else arith.to_twos(arith.clamp(val))
        )
        await Timer(2, unit="step")
        assert (
            dut.DOWNER_LIMIT.value,
            dut.UPPER_LIMIT.value,
            dut.DATA_VALID.value,
        ) == dtct_model(
            arith.clamp(val),
            bitwidth,
            num_adders,
            is_signed,
        )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [6, 10])
@pytest.mark.parametrize("num_adders", [2, 3, 4])
@pytest.mark.parametrize("is_signed", [0, 1])
def test_adder_range_violation(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_adders: int,
    is_signed: int,
):
    cocotb_test_fixture.set_top_module_name("ADDER_RANGE_VIOLATION")
    cocotb_test_fixture.run(
        params={"BITWIDTH": bitwidth, "NUM_ADDERS": num_adders, "IS_SIGNED": is_signed},
        defines={},
    )


@pytest.mark.simulation
@pytest.mark.parametrize("bitwidth", [8])
@pytest.mark.parametrize("num_adders", [3])
@pytest.mark.parametrize("is_signed", [0, 1])
def test_adder_range_violation_build(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    num_adders: int,
    is_signed: int,
):
    artifact_dir = cocotb_test_fixture.get_artifact_dir()
    build_dir = artifact_dir / "verilog"

    load_and_plugin(
        type="adder_range_violation",
        id=f"{bitwidth}",
        params={"BITWIDTH": bitwidth, "NUM_ADDERS": num_adders, "IS_SIGNED": is_signed},
        packages=["adders"],
        path2save=build_dir,
    )
    cocotb_test_fixture.set_top_module_name(f"ADDER_RANGE_VIOLATION_{bitwidth}")
    cocotb_test_fixture.set_srcs(collect_all_srcs_from_build_dir(build_dir))
    cocotb_test_fixture.run(params={}, defines={})
