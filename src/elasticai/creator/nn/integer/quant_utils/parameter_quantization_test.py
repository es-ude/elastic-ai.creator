import pytest
import torch

from . import MPQParams, QParams
from .Observers import GlobalMinMaxObserver
from .parameter_quantization import quantize_bias, quantize_weights


@pytest.fixture(params=[QParams, MPQParams], ids=["standard", "mixed_precision"])
def params_module(request):
    return request.param


def make_params(module, scheme):
    bits = torch.tensor(8) if module is MPQParams else 8
    return getattr(module, scheme)(bits, GlobalMinMaxObserver())


@pytest.mark.parametrize(
    "scheme,zero_point,expected",
    [
        ("SymmetricSignedQParams", 0, [-127, -2, 0, 2, 127]),
        ("AsymmetricSignedQParams", -128, [0, 0, 0, 2, 255]),
        ("AsymmetricUnsignedQParams", 0, [0, 0, 0, 2, 255]),
    ],
)
def test_weights_preserve_existing_quantization(
    params_module, scheme, zero_point, expected
):
    params = make_params(params_module, scheme)
    params.scale_factor.fill_(0.5)
    params.zero_point.fill_(zero_point)
    weights = torch.tensor([-200.0, -1.0, 0.0, 1.0, 200.0])
    result = quantize_weights(weights, params)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.int32))
    assert result.device.type == "cpu"


@pytest.mark.parametrize(
    "scheme,expected",
    [
        ("SymmetricSignedQParams", [-131071, 131071]),
        ("AsymmetricSignedQParams", [-131072, 131071]),
        ("AsymmetricUnsignedQParams", [0, 262143]),
    ],
)
def test_bias_saturates_at_the_requested_width(params_module, scheme, expected):
    params = make_params(params_module, scheme)
    params.zero_point.fill_(7)
    result = quantize_bias(torch.tensor([-0.5, 0.5]), params, torch.tensor(2.0**-20), 18)
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.int32))
    assert int(params.quant_bits) == 18
    assert params.scale_factor.item() == 2.0**-20
    assert params.zero_point.item() == 0
    assert params.observer.forward_num.item() == 0
    assert result.device.type == "cpu"


@pytest.mark.parametrize("bits", [21, 24, 32])
def test_bias_preserves_values_when_the_requested_width_fits(params_module, bits):
    params = make_params(params_module, "SymmetricSignedQParams")
    result = quantize_bias(torch.tensor([-0.5, 0.5]), params, torch.tensor(2.0**-20), bits)
    torch.testing.assert_close(result, torch.tensor([-524288, 524288], dtype=torch.int32))
    assert int(params.quant_bits) == bits


def test_32_bit_bias_saturates_before_integer_conversion(params_module):
    params = make_params(params_module, "SymmetricSignedQParams")
    result = quantize_bias(torch.tensor([-2.0**31, 2.0**31]), params, torch.tensor(1.0), 32)
    torch.testing.assert_close(result, torch.tensor([-2147483647, 2147483647], dtype=torch.int32))


def test_32_bit_bias_round_trip_does_not_overflow(params_module):
    params = make_params(params_module, "SymmetricSignedQParams")
    bias = torch.tensor([-0.5, 0.5])
    result = quantize_bias(bias, params, torch.tensor(2.0**-20), 32)
    torch.testing.assert_close(params.dequantize(result), bias)
