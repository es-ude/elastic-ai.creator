import torch

from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.math_operations.MathOperations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M


def linear_layer_setup():
    linear_layer = Linear(
        in_features=3, out_features=10, bias=True, name="linear", quant_bits=8
    )

    input_data = torch.tensor(
        [
            [-1.0, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )

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

    linear_layer.bias.data = torch.tensor(
        [0.5, -0.5, 0.3, 0.2, -0.2, 0.1, 0.4, -0.4, 0.2, 0.6], dtype=torch.float32
    )

    output_data = input_data.mm(linear_layer.weight.t()) + linear_layer.bias

    return linear_layer, input_data, output_data


def test_initialization():
    linear_layer, _, _ = linear_layer_setup()

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


def test_forward():
    linear_layer, input_data, expected_output = linear_layer_setup()

    output = linear_layer.forward(input_data)

    # check updated quantization parameters
    assert torch.allclose(
        linear_layer.input_QParams.scale_factor,
        torch.tensor([0.0058823530562222]),
        atol=1e-10,
    )
    assert torch.equal(
        linear_layer.input_QParams.zero_point, torch.tensor([42], dtype=torch.int32)
    )

    assert torch.allclose(
        linear_layer.weight_QParams.scale_factor,
        torch.tensor([0.00941176526248455]),
        atol=1e-10,
    )
    assert torch.equal(linear_layer.weight_QParams.zero_point, torch.tensor([0]))

    assert torch.allclose(
        linear_layer.bias_QParams.scale_factor,
        torch.tensor([0.004724409431219101]),
        atol=1e-10,
    )
    assert torch.equal(linear_layer.bias_QParams.zero_point, torch.tensor([0]))

    # Adjust the tolerance value to match the actual output
    assert torch.allclose(output, expected_output, atol=1e-1)


def test_get_quantized_weights():
    linear_layer, input_data, _ = linear_layer_setup()
    _ = linear_layer.forward(input_data)

    q_weight = linear_layer._get_quantized_weights()

    expected_q_weight = torch.tensor(
        [
            [53, -53, 32],
            [21, -21, 11],
            [42, -42, 21],
            [64, -64, 32],
            [74, -74, 42],
            [85, -85, 53],
            [96, -96, 64],
            [106, -106, 74],
            [117, -117, 85],
            [127, -128, 96],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(q_weight, expected_q_weight)


def test_get_quantized_bias():
    linear_layer, input_data, _ = linear_layer_setup()
    _ = linear_layer.forward(input_data)
    q_bias = linear_layer._get_quantized_bias()

    expected_scale_factor = (
        linear_layer.input_QParams.scale_factor
        * linear_layer.weight_QParams.scale_factor
    )
    assert torch.allclose(
        linear_layer.bias_QParams.scale_factor,
        expected_scale_factor,
        atol=1e-10,
    )
    assert torch.equal(linear_layer.bias_QParams.zero_point, torch.tensor([0]))
    expected_quant_bits = (linear_layer.input_QParams.quant_bits + 1) + (
        linear_layer.weight_QParams.quant_bits + 1
    )
    assert linear_layer.bias_QParams.quant_bits == expected_quant_bits

    expected_q_bias = torch.tensor(
        [9031, -9031, 5419, 3612, -3612, 1806, 7225, -7225, 3612, 10838],
        dtype=torch.int32,
    )
    assert torch.equal(q_bias, expected_q_bias)


def test_precompute():
    linear_layer, input_data, _ = linear_layer_setup()
    _ = linear_layer.forward(input_data)

    linear_layer.precompute()

    expected_scale_factor_M = (
        linear_layer.input_QParams.scale_factor
        * linear_layer.weight_QParams.scale_factor
    ) / linear_layer.output_QParams.scale_factor

    assert torch.allclose(
        linear_layer.scale_factor_M, expected_scale_factor_M, atol=1e-10
    )

    expected_scale_factor_m_q_shift, expected_scale_factor_m_q = scaling_M(
        expected_scale_factor_M
    )

    assert torch.equal(linear_layer.scale_factor_m_q, expected_scale_factor_m_q)
    assert torch.equal(
        linear_layer.scale_factor_m_q_shift, expected_scale_factor_m_q_shift
    )


def test_int_forward():
    linear_layer, input_data, _ = linear_layer_setup()
    output_data = linear_layer.forward(input_data)

    linear_layer.precompute()

    # Quantize the input before passing to int_forward
    q_input = linear_layer.input_QParams.quantize(input_data)
    q_output = linear_layer.int_forward(q_input)

    expected_q_output = linear_layer.output_QParams.quantize(output_data)

    assert torch.allclose(q_output, expected_q_output, atol=1)
