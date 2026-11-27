import importlib

import pytest
import torch

from . import MPQParams
from .Observers import GlobalMinMaxObserver
from .parameter_quantization import (
    AccumulatorQParams,
    accumulator_qparams,
    quantize_bias,
    quantize_weights,
)
from .SimQuant import SimQuant


@pytest.fixture(
    params=[importlib.import_module(".QParams", __package__), MPQParams],
    ids=["standard", "mixed_precision"],
)
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


def test_accumulator_derives_scale_and_width(params_module):
    inputs = make_params(params_module, "AsymmetricSignedQParams")
    weights = make_params(params_module, "SymmetricSignedQParams")
    inputs.scale_factor.fill_(0.25)
    weights.scale_factor.fill_(0.125)

    accumulator = accumulator_qparams(inputs, weights, (3,))
    assert accumulator.quant_bits == 19
    torch.testing.assert_close(accumulator.scale_factor, torch.tensor([0.03125]))

    inputs.scale_factor.fill_(0.5)
    torch.testing.assert_close(accumulator.scale_factor, torch.tensor([0.03125]))
    torch.testing.assert_close(
        accumulator_qparams(inputs, weights, (3,)).scale_factor, torch.tensor([0.0625])
    )


def test_bias_clips_to_accumulator_range():
    accumulator = AccumulatorQParams(18, torch.tensor(2.0**-20))
    result = quantize_bias(torch.tensor([-0.5, 0.5]), accumulator)
    torch.testing.assert_close(
        result, torch.tensor([-131071, 131071], dtype=torch.int32)
    )
    assert result.device.type == "cpu"


@pytest.mark.parametrize("accumulator_width", [21, 24, 31, 32])
def test_bias_quantization_matches_fake_quantization(accumulator_width):
    accumulator = AccumulatorQParams(accumulator_width, torch.tensor(2.0**-20))
    bias = torch.tensor([-0.5, 0.5], requires_grad=True)

    codes = quantize_bias(bias, accumulator)
    fake_quantized = SimQuant.apply(bias, accumulator)

    torch.testing.assert_close(
        codes, torch.tensor([-524288, 524288], dtype=torch.int32)
    )
    torch.testing.assert_close(fake_quantized, accumulator.dequantize(codes))
    fake_quantized.sum().backward()
    torch.testing.assert_close(bias.grad, torch.ones_like(bias))


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("bits", [6, 8, 16])
@pytest.mark.parametrize(
    "scheme",
    ["SymmetricSignedQParams", "AsymmetricSignedQParams", "AsymmetricUnsignedQParams"],
)
def test_low_bit_quantization_rounding_and_saturation(
    params_module, device, bits, scheme
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    params = make_params(params_module, scheme).to(device)
    params.set_quant_range(bits)
    params.scale_factor.fill_(1.0)
    params.zero_point.zero_()
    inputs = torch.tensor([-1e6, -1.6, 0.0, 1.6, 1e6], device=device)

    result = params.quantize(inputs)

    lower, upper = int(params.min_quant.item()), int(params.max_quant.item())
    expected = [max(lower, min(upper, v)) for v in [-1000000, -2, 0, 2, 1000000]]
    torch.testing.assert_close(result.cpu(), torch.tensor(expected, dtype=torch.int32))
    assert result.device.type == device
