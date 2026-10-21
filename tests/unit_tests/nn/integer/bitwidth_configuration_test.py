import pytest
import torch

from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d
from elasticai.creator.nn.integer.ffn.ffn import FeedForwardNetwork
from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.relu.relu import ReLU
from elasticai.creator.nn.integer.sequential.sequential import Sequential


@pytest.mark.parametrize("convolution", [False, True])
def test_separate_activation_and_weight_widths_in_integer_inference(convolution):
    widths = dict(activation_bits=10, weight_bits=4, bias_bits=24, bias=True)
    layer = (
        Conv1d(in_channels=2, out_channels=1, kernel_size=1, **widths)
        if convolution
        else Linear(in_features=2, out_features=1, **widths)
    ).eval()
    assert layer.inputs_QParams.quant_bits == 10
    assert layer.outputs_QParams.quant_bits == 10
    assert layer.weight_QParams.quant_bits == 4
    assert layer.bias_QParams.quant_bits == 24
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, -0.5]).reshape_as(layer.weight))
        layer.bias.zero_()
    layer.inputs_QParams.scale_factor.fill_(0.25)
    layer.inputs_QParams.zero_point.fill_(-3)
    layer.weight_QParams.scale_factor.fill_(0.25)
    layer.weight_QParams.zero_point.fill_(1)
    layer.outputs_QParams.scale_factor.fill_(0.125)
    layer.outputs_QParams.zero_point.fill_(5)
    inputs = torch.tensor([[0.5, -0.5]])
    if convolution:
        inputs = inputs.unsqueeze(-1)
    layer.precompute()
    expected = layer.outputs_QParams.quantize(layer(inputs, enable_simquant=False))
    torch.testing.assert_close(
        layer.int_forward(layer.inputs_QParams.quantize(inputs)), expected
    )


def test_composed_uniform_layer_forwards_both_widths():
    layer = FeedForwardNetwork(
        name="ffn", d_model=2, ffn_dim=4, activation_bits=10, weight_bits=4
    )
    for module in layer.modules():
        if hasattr(module, "inputs_QParams"):
            assert module.inputs_QParams.quant_bits == 10
        if hasattr(module, "outputs_QParams"):
            assert module.outputs_QParams.quant_bits == 10
        if hasattr(module, "weight_QParams"):
            assert module.weight_QParams.quant_bits == 4


def test_quant_bits_remains_the_fallback():
    layer = Linear(in_features=1, out_features=1, bias=True, quant_bits=8)
    assert layer.inputs_QParams.quant_bits == layer.outputs_QParams.quant_bits == 8
    assert layer.weight_QParams.quant_bits == 8
    override = Linear(
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
    widths = dict(activation_bits=10, weight_bits=4)
    first = Linear(in_features=1, out_features=1, bias=True, **widths)
    relu = ReLU(**widths)
    last = Linear(in_features=1, out_features=1, bias=True, **widths)
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
