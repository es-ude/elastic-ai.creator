import pytest

import elasticai.creator.nn.fixed_point as nn_creator
from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn import Sequential
from tests.integration_tests.nn.fixed_point.sequential_routine import (
    routine_testing_sequential_module,
)


@pytest.mark.simulation
@pytest.mark.parametrize(
    "total_bits, frac_bits, features_in, features_out",
    [
        (6, 4, 20, 12),
        (8, 6, 20, 8),
        (10, 8, 24, 20),
    ],
)
def test_build_test_batchnorm_linear(
    total_bits: int,
    frac_bits: int,
    features_in: int,
    features_out: int,
) -> None:
    file_name = (
        f"TestBatchNormLinear_{total_bits}_{frac_bits}_{features_in}x{features_out}"
    )
    fxp = FxpArithmetic(
        FxpParams(total_bits=total_bits, frac_bits=frac_bits, signed=True)
    )

    dut = Sequential(
        nn_creator.BatchNormedLinear(
            in_features=features_in,
            out_features=features_out,
            total_bits=total_bits,
            frac_bits=frac_bits,
        )
    )

    routine_testing_sequential_module(
        dut=dut,
        file_name=file_name,
        fxp=fxp,
        feat_in=features_in,
        check_quant=False,
    )
