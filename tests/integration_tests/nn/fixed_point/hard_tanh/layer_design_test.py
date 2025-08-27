import pytest

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from tests.integration_tests.nn.fixed_point.precomputed_routine import (
    routine_testing_precomputed_module,
)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps",
    [(6, 4, 32), (8, 4, 32), (10, 8, 64), (10, 9, 64)],
)
def test_build_test_hardtanh_design(
    total_bits: int, frac_bits: int, num_steps: int
) -> None:
    file_name = f"TestHardTanh_{total_bits}_{frac_bits}_{num_steps}"
    fxp = FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=frac_bits))

    dut = nn_creator.HardTanh(
        total_bits=total_bits,
        frac_bits=frac_bits,
        min_val=-1.0,
        max_val=1.0,
    )
    routine_testing_precomputed_module(
        dut=dut,
        num_steps=2 * num_steps,
        fxp=fxp,
        file_name=file_name,
        file_suffix="vhd",
    )
