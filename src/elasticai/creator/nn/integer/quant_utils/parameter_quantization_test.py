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
    "scheme",
    ["SymmetricSignedQParams", "AsymmetricSignedQParams", "AsymmetricUnsignedQParams"],
)
def test_bias_expands_width_to_preserve_values(params_module, scheme):
    params = make_params(params_module, scheme)
    params.zero_point.fill_(7)
    bias = torch.tensor([-0.5, 0.5] if params.is_signed else [0.0, 0.5])
    result = quantize_bias(bias, params, torch.tensor(2.0**-20))
    expected = [-524288, 524288] if params.is_signed else [0, 524288]
    torch.testing.assert_close(result, torch.tensor(expected, dtype=torch.int32))
    assert int(params.quant_bits) == (21 if params.is_signed else 20)
    assert params.scale_factor.item() == 2.0**-20
    assert params.zero_point.item() == 0
    assert params.observer.forward_num.item() == 0
    assert result.device.type == "cpu"


def test_bias_supports_wider_width_chosen_by_caller(params_module):
    params = make_params(params_module, "SymmetricSignedQParams")
    result = quantize_bias(
        torch.tensor([-0.5, 0.5]), params, torch.tensor(2.0**-20), 21
    )
    torch.testing.assert_close(
        result, torch.tensor([-524288, 524288], dtype=torch.int32)
    )
    assert int(params.quant_bits) == 21


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 2.0**31])
def test_bias_rejects_codes_that_cannot_be_stored(params_module, value):
    params = make_params(params_module, "SymmetricSignedQParams")
    with pytest.raises(ValueError):
        quantize_bias(torch.tensor([value]), params, torch.tensor(1.0), 18)


def test_unsigned_bias_rejects_negative_values(params_module):
    params = make_params(params_module, "AsymmetricUnsignedQParams")
    with pytest.raises(ValueError, match="negatives"):
        quantize_bias(torch.tensor([-1.0]), params, torch.tensor(1.0), 18)


@pytest.mark.parametrize(
    "values,bits", [([0.0], 2), ([-1.0, 1.0], 2), ([-8.0, 8.0], 5)]
)
def test_bias_width_is_selected_without_layer_hint(params_module, values, bits):
    params = make_params(params_module, "SymmetricSignedQParams")
    result = quantize_bias(torch.tensor(values), params, torch.tensor(1.0))
    assert int(params.quant_bits) == bits
    torch.testing.assert_close(result, torch.tensor(values, dtype=torch.int32))
