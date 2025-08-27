import pytest
import torch

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.arithmetic import FxpArithmetic, FxpParams
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point.math_operations import (
    MathOperations,
)
from tests.integration_tests.nn.fixed_point.sequential_routine import (
    routine_testing_sequential_module,
)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, features_in, features_out",
    [
        (4, 2, 12, 6),
        (10, 8, 24, 20),
    ],
)
def test_build_test_linear_hardtanh(
    total_bits: int,
    frac_bits: int,
    features_in: int,
    features_out: int,
) -> None:
    file_name = (
        f"TestLinearHardTanh_{total_bits}_{frac_bits}_{features_in}x{features_out}"
    )
    fxp = FxpArithmetic(FxpParams(total_bits=total_bits, frac_bits=frac_bits))
    math = MathOperations(fxp)

    dut = Sequential(
        nn_creator.Linear(
            in_features=features_in,
            out_features=features_out,
            total_bits=total_bits,
            frac_bits=frac_bits,
        ),
        nn_creator.HardTanh(total_bits=total_bits, frac_bits=frac_bits),
    )
    # --- Adapting values
    scale_amp = (fxp.maximum_as_rational - fxp.minimum_as_rational) / (2 * features_in)
    scale_min = -scale_amp / 2

    dut[0].weight.data = torch.nn.Parameter(
        math.quantize(
            scale_amp
            * torch.rand(
                size=(features_out, features_in),
            )
            + scale_min
            + torch.randint(low=-1, high=+1, size=(features_out, features_in))
            * fxp.config.minimum_step_as_rational
        )
    )
    dut[0].bias.data = torch.nn.Parameter(
        math.quantize(
            scale_amp
            * torch.rand(
                size=(features_out,),
            )
            + scale_min
            + torch.randint(low=-1, high=+1, size=(features_out,))
            * fxp.config.minimum_step_as_rational
        )
    )

    routine_testing_sequential_module(
        dut=dut,
        file_name=file_name,
        fxp=fxp,
        feat_in=features_in,
    )
