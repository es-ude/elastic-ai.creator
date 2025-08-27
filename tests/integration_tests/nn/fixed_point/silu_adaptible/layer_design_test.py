import pytest

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from tests.integration_tests.nn.fixed_point.precomputed_routine import (
    routine_testing_precomputed_module,
)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, num_steps", [(6, 4, 32), (8, 4, 32), (10, 9, 64)]
)
def test_build_test_silu_adapt_design(
    total_bits: int, frac_bits: int, num_steps: int
) -> None:
    file_name = f"TestAdaptSiLU_{total_bits}_{frac_bits}_{num_steps}"
    fxp = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )

    dut = nn_creator.AdaptableSiLU(
        total_bits=total_bits,
        frac_bits=frac_bits,
        num_steps=num_steps,
        sampling_intervall=(
            fxp.config.minimum_as_rational,
            fxp.config.maximum_as_rational,
        ),
    )
    routine_testing_precomputed_module(
        dut=dut,
        num_steps=2 * num_steps,
        fxp=fxp,
        file_name=file_name,
        file_suffix="vhd",
    )
