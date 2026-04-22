import pytest
import torch

from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
)
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant


@pytest.fixture
def setUp():
    x_r = torch.tensor([0.5, -1.5, 3.0, 2.5, -0.5], dtype=torch.float32)
    grad_output = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    min_quant = torch.tensor([-128], dtype=torch.int32)
    max_quant = torch.tensor([127], dtype=torch.int32)
    min_float = torch.tensor([-1.5], dtype=torch.float32)
    max_float = torch.tensor([3.0], dtype=torch.float32)

    scale_factor, zero_point, _, _ = calculate_asymmetric_quant_params(
        min_float, max_float, min_quant, max_quant, torch.tensor([1e-6])
    )
    # scale_factor = torch.tensor([0.0176], dtype=torch.float32)
    # zero_point = torch.tensor([-85], dtype=torch.int32)

    x_r_QParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver)
    x_r_QParams.min_float = min_float
    x_r_QParams.max_float = max_float
    x_r_QParams.scale_factor = scale_factor
    x_r_QParams.zero_point = zero_point

    return x_r, grad_output, x_r_QParams


def test_forward(setUp):
    x_r, _, x_r_QParams = setUp
    actual_result = SimQuant.apply(x_r, x_r_QParams)
    expected_result = x_r_QParams.dequantize(x_r_QParams.quantize(x_r))
    assert torch.allclose(actual_result, expected_result, rtol=1e-5, atol=1e-8)


def test_backward(setUp):
    x_r, grad_output, x_r_QParams = setUp
    x_r = x_r.clone().requires_grad_(True)
    result = SimQuant.apply(x_r, x_r_QParams)
    result.backward(grad_output)
    actual_result = x_r.grad

    expected_result = grad_output.clone()
    expected_result[x_r.gt(x_r_QParams.max_float)] = 0
    expected_result[x_r.lt(x_r_QParams.min_float)] = 0
    assert torch.allclose(actual_result, expected_result, rtol=1e-5, atol=1e-8)
