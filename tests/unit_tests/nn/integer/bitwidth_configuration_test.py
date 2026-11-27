import pytest
import torch

from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d
from elasticai.creator.nn.integer.ffn.ffn import FeedForwardNetwork
from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.relu.relu import ReLU
from elasticai.creator.nn.integer.sequential.sequential import Sequential


@pytest.mark.parametrize("convolution", [False, True])
def test_separate_activation_and_weight_widths_in_integer_inference(convolution):
    widths = dict(activation_bits=10, weight_bits=4, quant_bits=8)
    layer = (
        Conv1d(
            name="conv",
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            padding=0,
            seq_len=1,
            **widths,
        )
        if convolution
        else Linear(name="linear", in_features=2, out_features=1, bias=True, **widths)
    ).eval()
    assert layer.inputs_QParams.quant_bits == 10
    assert layer.outputs_QParams.quant_bits == 10
    assert layer.weight_QParams.quant_bits == 4
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, -0.5]).reshape_as(layer.weight))
        layer.bias.zero_()
    layer.inputs_QParams.scale_factor.fill_(0.25)
    layer.inputs_QParams.zero_point.fill_(-3)
    layer.weight_QParams.scale_factor.fill_(0.25)
    layer.weight_QParams.zero_point.fill_(1)
    layer.outputs_QParams.scale_factor.fill_(0.125)
    layer.outputs_QParams.zero_point.fill_(5)
    # With input scale 0.25 and zero point -3, these codes represent [0.5, -0.5].
    input_codes = torch.tensor([[-1, -5]], dtype=torch.int32)
    # The dot product is 0.5, which maps to output code 0.5 / 0.125 + 5 = 9.
    expected_codes = torch.tensor([[9]], dtype=torch.int32)
    if convolution:
        input_codes = input_codes.unsqueeze(-1)
        expected_codes = expected_codes.unsqueeze(-1)
    layer.precompute()
    assert layer.bias_QParams.quant_bits == 17
    torch.testing.assert_close(layer.int_forward(input_codes), expected_codes)


def test_composed_uniform_layer_forwards_both_widths():
    layer = FeedForwardNetwork(
        name="ffn",
        d_model=2,
        ffn_dim=4,
        num_dimensions=1,
        quant_bits=8,
        activation_bits=10,
        weight_bits=4,
    )
    for module in layer.modules():
        if hasattr(module, "inputs_QParams"):
            assert module.inputs_QParams.quant_bits == 10
        if hasattr(module, "outputs_QParams"):
            assert module.outputs_QParams.quant_bits == 10
        if hasattr(module, "weight_QParams"):
            assert module.weight_QParams.quant_bits == 4


def test_quant_bits_remains_the_fallback():
    layer = Linear(
        name="linear", in_features=1, out_features=1, bias=True, quant_bits=8
    )
    assert layer.inputs_QParams.quant_bits == layer.outputs_QParams.quant_bits == 8
    assert layer.weight_QParams.quant_bits == 8
    override = Linear(
        name="linear",
        in_features=1,
        out_features=1,
        bias=True,
        quant_bits=8,
        activation_bits=10,
        weight_bits=4,
    )
    assert (
        override.inputs_QParams.quant_bits == override.outputs_QParams.quant_bits == 10
    )
    assert override.weight_QParams.quant_bits == 4


def test_uniform_network_keeps_existing_activation_parameter_inheritance():
    widths = dict(activation_bits=10, weight_bits=4, quant_bits=8)
    first = Linear(name="linear", in_features=1, out_features=1, bias=True, **widths)
    relu = ReLU(name="relu", **widths)
    last = Linear(name="last", in_features=1, out_features=1, bias=True, **widths)
    with torch.no_grad():
        first.weight.fill_(0.5)
        last.weight.fill_(0.5)
        first.bias.zero_()
        last.bias.zero_()
    model = Sequential(first, relu, last, name="uniform", quant_data_dir=None)
    inputs = torch.tensor([[-1.0], [0.0], [1.0]])
    model(inputs)
    assert relu.inputs_QParams is first.outputs_QParams
    assert last.inputs_QParams is relu.outputs_QParams
    model.eval()
    model.precompute()
    codes = model.quantize_inputs(inputs)
    expected = last.int_forward(relu.int_forward(first.int_forward(codes)))
    torch.testing.assert_close(model.int_forward(codes), expected)
