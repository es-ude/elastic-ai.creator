import pytest

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from tests.integration_tests.nn.fixed_point.precomputed_routine import (
    routine_testing_precomputed_module,
)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, init", [(6, 4, 0.125), (8, 4, 0.0625), (10, 9, 0.03125)]
)
def test_build_test_prelu_design(total_bits: int, frac_bits: int, init: float) -> None:
    file_name = f"TestPReLU_{total_bits}_{frac_bits}"
    fxp = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )

    dut = nn_creator.PReLU(total_bits=total_bits, frac_bits=frac_bits, init=init)
    routine_testing_precomputed_module(
        dut=dut,
        num_steps=2**total_bits,
        fxp=fxp,
        file_name=file_name,
        file_suffix="vhd",
    )
