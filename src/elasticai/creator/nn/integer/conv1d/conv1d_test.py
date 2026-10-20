import pytest
import torch
from torch.nn import functional as F

from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d


@pytest.mark.parametrize("bias", [False, True])
def test_conv1d_bias_option_in_float_and_integer_paths(bias):
    layer = Conv1d(
        in_channels=1, out_channels=1, kernel_size=1, quant_bits=8, bias=bias
    )
    with torch.no_grad():
        layer.weight.fill_(2)
        if layer.bias is not None:
            layer.bias.fill_(3)
    inputs = torch.tensor([[[1.0, 2.0, 3.0]]])
    layer.eval()

    assert (layer.bias is not None) == bias
    expected = F.conv1d(inputs, layer.weight, layer.bias)
    torch.testing.assert_close(layer(inputs, enable_simquant=False), expected)
    torch.testing.assert_close(layer(inputs), expected)

    layer.precompute()
    assert (layer.q_bias is not None) == bias
    torch.testing.assert_close(
        layer.int_forward(inputs.to(torch.int32)), expected.to(torch.int32)
    )


@pytest.mark.parametrize(
    "options",
    [
        {"stride": 2},
        {"stride": (2,)},
        {"dilation": 2},
        {"groups": 2},
        {"padding_mode": "reflect"},
    ],
)
def test_conv1d_rejects_unsupported_options(options):
    with pytest.raises(ValueError, match="only supports"):
        Conv1d(in_channels=2, out_channels=2, kernel_size=3, quant_bits=8, **options)


def test_conv1d_accepts_unit_tuple_options():
    layer = Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        quant_bits=8,
        stride=(1,),
        dilation=(1,),
    )
    assert layer.stride == (1,)
    assert layer.dilation == (1,)


@pytest.mark.parametrize("kernel_size,output_shift", [(50, 17), (20000, 25)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("negative_inputs", [False, True])
def test_conv1d_preserves_accumulator_until_requantization(
    kernel_size, output_shift, bias, negative_inputs
):
    layer = Conv1d(
        in_channels=2,
        out_channels=1,
        kernel_size=kernel_size,
        quant_bits=8,
        bias=bias,
    ).eval()
    with torch.no_grad():
        layer.weight.fill_(255)
        if layer.bias is not None:
            layer.bias.zero_()

    # Unit input/weight scales and opposite-end zero points give centered
    # operands of magnitude 255, while all stored codes remain valid int8.
    layer.weight_QParams.zero_point.fill_(-128)
    layer.inputs_QParams.zero_point.fill_(127 if negative_inputs else -128)
    layer.outputs_QParams.scale_factor.fill_(2**output_shift)
    layer.precompute()

    inputs = torch.full(
        (1, 2, kernel_size), -128 if negative_inputs else 127, dtype=torch.int32
    )
    accumulator = 2 * kernel_size * 255 * 255 * (-1 if negative_inputs else 1)
    expected_code = round(accumulator / (2**output_shift))
    # Requantization keeps the final code in range, so output saturation
    # cannot hide premature accumulator clipping or int32 overflow.
    assert -128 < expected_code < 127
    torch.testing.assert_close(
        layer.int_forward(inputs),
        torch.tensor([[[expected_code]]], dtype=torch.int32),
    )


@pytest.mark.parametrize("bias_value", [-0.5, 0.5])
def test_conv1d_preserves_bias_larger_than_product_range(bias_value):
    layer = Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        quant_bits=8,
        bias=True,
    ).eval()
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(bias_value)
    layer.inputs_QParams.scale_factor.fill_(2.0**-10)
    layer.weight_QParams.scale_factor.fill_(2.0**-10)
    layer.outputs_QParams.scale_factor.fill_(2.0**-7)
    layer.precompute()

    torch.testing.assert_close(
        layer.int_forward(torch.zeros((1, 1, 1), dtype=torch.int32)),
        torch.tensor([[[int(bias_value * 128)]]], dtype=torch.int32),
    )
