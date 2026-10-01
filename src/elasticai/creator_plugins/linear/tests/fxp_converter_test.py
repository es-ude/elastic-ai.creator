import cocotb
import numpy as np
import pytest
from cocotb.triggers import ReadOnly, Timer

from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.testing import CocotbTestFixture, eai_testbench
from elasticai.creator_plugins import linear


def model(x: int, bitwidth: int, frac_width: int) -> int:
    y = x >> frac_width
    lo = -(1 << (bitwidth - 1))
    hi = (1 << (bitwidth - 1)) - 1
    return max(lo, min(hi, y))


@cocotb.test()
@eai_testbench
async def fxp_converter_test(
    dut,
    bitwidth: int,
    frac_width: int,
    data_in: list[list[int]],
    expected_in: list[list[int]],
):
    dut.DATA_IN.value = 0

    for data_batch, expected_batch in zip(data_in, expected_in):
        dut.DATA_IN.value = data_batch
        await Timer(time=4, unit="ns")

        await ReadOnly()
        assert dut.DATA_OUT.value.to_signed() == expected_batch
        await Timer(time=4, unit="ns")


@pytest.mark.simulation
@pytest.mark.parametrize(
    "bitwidth, frac_width",
    [
        (4, 0),
        (8, 4),
        (8, 8),
        (12, 1),
        (12, 11),
    ],
)
def test_template(
    cocotb_test_fixture: CocotbTestFixture,
    bitwidth: int,
    frac_width: int,
) -> None:
    arith = FxpArithmetic(
        FxpParams(total_bits=2 * bitwidth, frac_bits=2 * frac_width, signed=True)
    )

    data_in = np.random.randint(
        low=arith.minimum_as_integer,
        high=arith.maximum_as_integer,
        size=2 ** (bitwidth - 1),
    ).tolist()
    data_in.extend([arith.minimum_as_integer, arith.maximum_as_integer])
    data_in.extend([arith.minimum_as_integer, arith.maximum_as_integer])
    expected_in = [model(val, bitwidth, frac_width) for val in data_in]
    cocotb_test_fixture.write(
        {
            "data_in": data_in,
            "expected_in": expected_in,
        }
    )
    cocotb_test_fixture.set_top_module_name("FXP_DOWNCONVERTER")
    cocotb_test_fixture.clear_srcs()
    cocotb_test_fixture.add_srcs_from_package(linear, "verilog/fxp_converter.v")
    cocotb_test_fixture.run(
        params={
            "BITWIDTH": bitwidth,
            "FRAC_WIDTH": frac_width,
        },
        defines={},
    )
