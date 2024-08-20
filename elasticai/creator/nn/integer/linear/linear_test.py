import torch

from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.math_operations.MathOperations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver


def linear_layer_setup():
    return Linear(
        in_features=3, out_features=10, bias=True, name="linear", quant_bits=8
    )


def test_initialization():
    linear_layer = linear_layer_setup()
    assert linear_layer.in_features == 3
    assert linear_layer.out_features == 10
    assert linear_layer.bias is not None
    assert linear_layer.name == "linear"
    assert linear_layer.quant_bits == 8
    assert linear_layer.weight is not None
    assert linear_layer.weight.shape == (10, 3)
    assert linear_layer.bias.shape == (10,)

    assert linear_layer.weight_QParams.quant_bits == 8
    assert isinstance(linear_layer.weight_QParams.observer, GlobalMinMaxObserver)
    assert linear_layer.bias_QParams.quant_bits == 8
    assert isinstance(linear_layer.bias_QParams.observer, GlobalMinMaxObserver)
    assert linear_layer.input_QParams.quant_bits == 8
    assert isinstance(linear_layer.input_QParams.observer, GlobalMinMaxObserver)
    assert linear_layer.output_QParams.quant_bits == 8
    assert isinstance(linear_layer.output_QParams.observer, GlobalMinMaxObserver)
    assert isinstance(linear_layer.math_ops, MathOperations)


def test_get_quantized_weights():
    linear_layer = linear_layer_setup()

    linear_layer.weight.data = torch.tensor(
        [
            [0.5, -0.5, 0.3],
            [0.2, -0.2, 0.1],
            [0.4, -0.4, 0.2],
            [0.6, -0.6, 0.3],
            [0.7, -0.7, 0.4],
            [0.8, -0.8, 0.5],
            [0.9, -0.9, 0.6],
            [1.0, -1.0, 0.7],
            [1.1, -1.1, 0.8],
            [1.2, -1.2, 0.9],
        ],
        dtype=torch.float32,
    )
    linear_layer.weight_QParams.update_quant_params(linear_layer.weight)
    q_weight = linear_layer._get_quantized_weights()
    assert q_weight is not None
    assert q_weight.shape == (10, 3)
    assert q_weight.dtype == torch.int32


def test_get_quantized_bias():
    linear_layer = linear_layer_setup()

    linear_layer.bias.data = torch.tensor(
        [0.5, -0.5, 0.3, 0.2, -0.2, 0.1, 0.4, -0.4, 0.2, 0.6], dtype=torch.float32
    )
    linear_layer.bias_QParams.update_quant_params(linear_layer.bias)
    q_bias = linear_layer._get_quantized_bias()
    assert q_bias is not None
    assert q_bias.shape == (10,)
    assert q_bias.dtype == torch.int32
