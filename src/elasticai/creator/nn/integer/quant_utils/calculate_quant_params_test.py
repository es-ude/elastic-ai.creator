import pytest
import torch

from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
    calculate_symmetric_quant_params,
)

EPS = torch.tensor(torch.finfo(torch.float32).eps)


def _args(min_float, max_float, min_quant, max_quant, dtype=torch.float32):
    return dict(
        min_float=torch.tensor([min_float], dtype=dtype),
        max_float=torch.tensor([max_float], dtype=dtype),
        min_quant=torch.tensor([min_quant], dtype=torch.int32),
        max_quant=torch.tensor([max_quant], dtype=torch.int32),
        eps=EPS,
    )


def test_asymmetric_maps_float_range_onto_quant_range():
    scale_factor, zero_point, min_float, max_float = calculate_asymmetric_quant_params(
        **_args(-2.0, 3.0, -128, 127)
    )

    assert torch.equal(scale_factor, torch.tensor([5 / 255], dtype=torch.float32))
    assert zero_point.item() == -26  # 127 - 3 / scale_factor
    assert min_float.item() == -2.0
    assert max_float.item() == 3.0


def test_asymmetric_clamps_zero_point_to_quant_range():
    _, zero_point, _, _ = calculate_asymmetric_quant_params(**_args(1.0, 3.0, 0, 255))

    assert zero_point.item() == 0


def test_symmetric_widens_float_range_and_has_zero_point_zero():
    scale_factor, zero_point, min_float, max_float = calculate_symmetric_quant_params(
        **_args(-2.0, 3.0, -127, 127)
    )

    assert torch.equal(scale_factor, torch.tensor([6 / 254], dtype=torch.float32))
    assert zero_point.item() == 0
    assert min_float.item() == -3.0
    assert max_float.item() == 3.0


@pytest.mark.parametrize(
    "calculate", [calculate_asymmetric_quant_params, calculate_symmetric_quant_params]
)
def test_results_have_fixed_dtypes(calculate):
    scale_factor, zero_point, _, _ = calculate(
        **_args(-1.0, 2.0, -128, 127, dtype=torch.float64)
    )

    assert scale_factor.dtype == torch.float32
    assert zero_point.dtype == torch.int32


@pytest.mark.parametrize(
    "calculate", [calculate_asymmetric_quant_params, calculate_symmetric_quant_params]
)
def test_constant_zero_uses_eps_as_scale_factor(calculate):
    scale_factor, _, _, _ = calculate(**_args(0.0, 0.0, -128, 127))

    assert scale_factor.item() == EPS.item()


@pytest.mark.parametrize(
    "calculate", [calculate_asymmetric_quant_params, calculate_symmetric_quant_params]
)
@pytest.mark.parametrize(
    "argument, value",
    [
        ("min_float", torch.tensor([-1], dtype=torch.int32)),
        ("max_float", torch.tensor([1], dtype=torch.int32)),
        ("min_quant", torch.tensor([-128.0])),
        ("max_quant", torch.tensor([127.0])),
        ("eps", torch.tensor(1, dtype=torch.int32)),
        ("min_float", -1.0),
    ],
)
def test_rejects_arguments_of_wrong_dtype(calculate, argument, value):
    arguments = _args(-1.0, 1.0, -128, 127) | {argument: value}

    with pytest.raises(TypeError, match=argument):
        calculate(**arguments)
