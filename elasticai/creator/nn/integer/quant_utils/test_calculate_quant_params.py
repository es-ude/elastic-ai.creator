import pytest
import torch

from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
    calculate_symmetric_quant_params,
)


@pytest.fixture
def eps():
    return torch.tensor(torch.finfo(torch.float32).eps)


@pytest.fixture
def asymmetric_signed_setup(eps):
    inputs = torch.tensor([-3.2, 2.5], dtype=torch.float32)
    min_float = torch.min(inputs)
    max_float = torch.max(inputs)

    min_quant = torch.tensor([-128], dtype=torch.int32)
    max_quant = torch.tensor([127], dtype=torch.int32)

    return min_float, max_float, min_quant, max_quant


@pytest.fixture
def asymmetric_unsigned_setup():
    inputs = torch.tensor([-3.2, 2.5], dtype=torch.float32)
    min_float = torch.min(inputs)
    max_float = torch.max(inputs)

    min_quant = torch.tensor([0], dtype=torch.int32)
    max_quant = torch.tensor([255], dtype=torch.int32)

    return min_float, max_float, min_quant, max_quant


@pytest.fixture
def symmetric_signed_setup():
    inputs = torch.tensor([-1.002, 1.0], dtype=torch.float32)
    min_float = torch.min(inputs)
    max_float = torch.max(inputs)
    min_quant = torch.tensor([-127], dtype=torch.int32)
    max_quant = torch.tensor([127], dtype=torch.int32)

    return min_float, max_float, min_quant, max_quant


@pytest.fixture
def symmetric_unsigned_setup():
    inputs = torch.tensor([-1.002, 1.0], dtype=torch.float32)
    min_float = torch.min(inputs)
    max_float = torch.max(inputs)

    min_quant = torch.tensor([0], dtype=torch.int32)
    max_quant = torch.tensor([254], dtype=torch.int32)

    return min_float, max_float, min_quant, max_quant


def test_calculate_asymmetric_quant_params_signed(asymmetric_signed_setup, eps):
    min_float, max_float, min_quant, max_quant = asymmetric_signed_setup

    expected_scale_factor = (max_float - min_float) / (
        max_quant.float() - min_quant.float()
    )
    expected_scale_factor = torch.max(expected_scale_factor, eps)
    expected_scale_factor = expected_scale_factor.clone().detach().to(torch.float32)

    expected_zero_point = max_quant - (max_float / expected_scale_factor)
    expected_zero_point = expected_zero_point.round_().clamp(min_quant, max_quant)
    expected_zero_point = expected_zero_point.clone().detach().to(torch.int32)

    expected_min_float = min_float
    expected_max_float = max_float

    actual_scale_factor, actual_zero_point, actual_min_float, actual_max_float = (
        calculate_asymmetric_quant_params(
            min_float, max_float, min_quant, max_quant, eps
        )
    )
    assert torch.isclose(actual_scale_factor, expected_scale_factor, atol=1e-10)
    assert torch.equal(actual_zero_point, expected_zero_point)
    assert torch.equal(actual_min_float, expected_min_float)
    assert torch.equal(actual_max_float, expected_max_float)


def test_calculate_asymmetric_quant_params_unsigned(asymmetric_unsigned_setup, eps):
    min_float, max_float, min_quant, max_quant = asymmetric_unsigned_setup

    expected_scale_factor = (max_float - min_float) / (
        max_quant.float() - min_quant.float()
    )
    expected_scale_factor = torch.max(expected_scale_factor, eps)
    expected_scale_factor = expected_scale_factor.clone().detach().to(torch.float32)

    expected_zero_point = max_quant - (max_float / expected_scale_factor)
    expected_zero_point = expected_zero_point.round_().clamp(min_quant, max_quant)
    expected_zero_point = expected_zero_point.clone().detach().to(torch.int32)

    expected_min_float = min_float
    expected_max_float = max_float

    actual_scale_factor, actual_zero_point, actual_min_float, actual_max_float = (
        calculate_asymmetric_quant_params(
            min_float, max_float, min_quant, max_quant, eps
        )
    )
    assert torch.isclose(actual_scale_factor, expected_scale_factor, atol=1e-10)
    assert torch.equal(actual_zero_point, expected_zero_point)
    assert torch.equal(actual_min_float, expected_min_float)
    assert torch.equal(actual_max_float, expected_max_float)


def test_calculate_symmetric_quant_params_signed(symmetric_signed_setup, eps):
    min_float, max_float, min_quant, max_quant = symmetric_signed_setup

    max_extent = torch.max(torch.abs(min_float), torch.abs(max_float))
    max_float = max_extent
    min_float = -max_extent

    expected_scale_factor = (max_float - min_float) / (
        max_quant.float() - min_quant.float()
    )
    expected_scale_factor = torch.max(expected_scale_factor, eps)

    expected_zero_point = torch.zeros(expected_scale_factor.size())
    expected_zero_point = expected_zero_point.clone().detach().to(torch.int32)

    expected_min_float = min_float
    expected_max_float = max_float

    actual_scale_factor, actual_zero_point, actual_min_float, actual_max_float = (
        calculate_symmetric_quant_params(
            min_quant, max_quant, min_float, max_float, eps
        )
    )
    assert torch.isclose(actual_scale_factor, expected_scale_factor, atol=1e-10)
    assert torch.equal(actual_zero_point, expected_zero_point)
    assert torch.equal(actual_min_float, expected_min_float)
    assert torch.equal(actual_max_float, expected_max_float)


def test_calculate_symmetric_quant_params_unsigned(symmetric_unsigned_setup, eps):
    min_float, max_float, min_quant, max_quant = symmetric_unsigned_setup

    max_extent = torch.max(torch.abs(min_float), torch.abs(max_float))
    max_float = max_extent
    min_float = -max_extent

    expected_scale_factor = (max_float - min_float) / (
        max_quant.float() - min_quant.float()
    )
    expected_scale_factor = torch.max(expected_scale_factor, eps)

    expected_zero_point = torch.zeros(expected_scale_factor.size())
    expected_zero_point = expected_zero_point.clone().detach().to(torch.int32)

    expected_min_float = min_float
    expected_max_float = max_float

    actual_scale_factor, actual_zero_point, actual_min_float, actual_max_float = (
        calculate_symmetric_quant_params(
            min_quant, max_quant, min_float, max_float, eps
        )
    )
    assert torch.isclose(actual_scale_factor, expected_scale_factor, atol=1e-10)
    assert torch.equal(actual_zero_point, expected_zero_point)
    assert torch.equal(actual_min_float, expected_min_float)
    assert torch.equal(actual_max_float, expected_max_float)
