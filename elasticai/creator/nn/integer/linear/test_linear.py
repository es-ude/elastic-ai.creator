import pytest
import torch
from torch.nn import functional as F

from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.math_operations.MathOperations import MathOperations
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant

inputs = torch.tensor(
    [
        [-1.0, 0.5, 0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.0, 0.5],
    ],
    dtype=torch.float32,
)


@pytest.fixture
def linear_layer() -> Linear:
    linear_layer = Linear(
        name="linear", in_features=3, out_features=10, bias=True, quant_bits=8
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

    return linear_layer


def test_linear_layer_initialization_weight_quant_bits(linear_layer) -> None:
    assert linear_layer.weight_QParams.quant_bits == 8


def test_linear_layer_initialization_bias_quant_bits(linear_layer) -> None:
    assert linear_layer.bias_QParams.quant_bits == 8


def test_linear_layer_initialization_input_quant_bits(linear_layer) -> None:
    assert linear_layer.inputs_QParams.quant_bits == 8


def test_linear_layer_initialization_output_quant_bits(linear_layer) -> None:
    assert linear_layer.outputs_QParams.quant_bits == 8


def test_linear_layer_initialization_weight_observer(linear_layer) -> None:
    assert isinstance(linear_layer.weight_QParams.observer, GlobalMinMaxObserver)


def test_linear_layer_initialization_bias_observer(linear_layer) -> None:
    assert isinstance(linear_layer.bias_QParams.observer, GlobalMinMaxObserver)


def test_linear_layer_initialization_input_observer(linear_layer) -> None:
    assert isinstance(linear_layer.inputs_QParams.observer, GlobalMinMaxObserver)


def test_linear_layer_initialization_output_observer(linear_layer) -> None:
    assert isinstance(linear_layer.outputs_QParams.observer, GlobalMinMaxObserver)


def test_linear_layer_initialization_math_ops(linear_layer) -> None:
    assert isinstance(linear_layer.math_ops, MathOperations)


def test_not_update_quant_params_of_inputs_QParams_in_forward(linear_layer) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.inputs_QParams.min_float == torch.ones((1), dtype=torch.float32)
    assert linear_layer.inputs_QParams.max_float == torch.ones((1), dtype=torch.float32)
    assert linear_layer.inputs_QParams.scale_factor == torch.ones(
        (1), dtype=torch.float32
    )
    assert linear_layer.inputs_QParams.zero_point == torch.zeros((1), dtype=torch.int32)


def test_use_given_inputs_QParams_in_forward(linear_layer) -> None:
    linear_layer.train()
    given_inputs_QParams = AsymmetricSignedQParams(
        quant_bits=6, observer=GlobalMinMaxObserver()
    )
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.inputs_QParams == given_inputs_QParams


def test_update_quant_params_of_inputs_QParams_in_forward(linear_layer) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.inputs_QParams.min_float == torch.tensor(
        [-1.0], dtype=torch.float32
    )
    assert linear_layer.inputs_QParams.max_float == torch.tensor(
        [0.5], dtype=torch.float32
    )
    assert torch.allclose(
        linear_layer.inputs_QParams.scale_factor,
        torch.tensor([0.0058823530562222], dtype=torch.float32),
        atol=1e-10,
    )
    assert torch.equal(
        linear_layer.inputs_QParams.zero_point, torch.tensor([42], dtype=torch.int32)
    )


def test_not_update_quant_params_of_weight_QParams_in_forward(linear_layer) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.weight_QParams.min_float == torch.ones((1), dtype=torch.float32)
    assert linear_layer.weight_QParams.max_float == torch.ones((1), dtype=torch.float32)
    assert linear_layer.weight_QParams.scale_factor == torch.ones(
        (1), dtype=torch.float32
    )
    assert linear_layer.weight_QParams.zero_point == torch.zeros((1), dtype=torch.int32)


def test_update_quant_params_of_weight_QParams_in_forward(linear_layer) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.weight_QParams.min_float == torch.tensor(
        [-1.2], dtype=torch.float32
    )
    assert linear_layer.weight_QParams.max_float == torch.tensor(
        [1.2], dtype=torch.float32
    )
    assert torch.allclose(
        linear_layer.weight_QParams.scale_factor,
        torch.tensor([0.00941176526248455], dtype=torch.float32),
        atol=1e-10,
    )
    assert torch.equal(linear_layer.weight_QParams.zero_point, torch.tensor([0]))


def test_not_update_quant_params_of_bias_QParams_in_forward(linear_layer) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.bias_QParams.min_float == torch.ones((1), dtype=torch.float32)
    assert linear_layer.bias_QParams.max_float == torch.ones((1), dtype=torch.float32)
    assert linear_layer.bias_QParams.scale_factor == torch.ones(
        (1), dtype=torch.float32
    )
    assert linear_layer.bias_QParams.zero_point == torch.zeros((1), dtype=torch.int32)


def test_update_quant_params_of_bias_QParams_in_forward(linear_layer) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.bias_QParams.min_float == torch.tensor(
        [-0.6], dtype=torch.float32  # because applied symmetric quantization
    )
    assert linear_layer.bias_QParams.max_float == torch.tensor(
        [0.6], dtype=torch.float32
    )
    assert torch.allclose(
        linear_layer.bias_QParams.scale_factor,
        torch.tensor([0.004724409431219101], dtype=torch.float32),
        atol=1e-10,
    )
    assert torch.equal(linear_layer.bias_QParams.zero_point, torch.tensor([0]))


def test_not_update_quant_params_of_outputs_QParams_in_forward(linear_layer) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)
    assert linear_layer.outputs_QParams.min_float == torch.ones(
        (1), dtype=torch.float32
    )
    assert linear_layer.outputs_QParams.max_float == torch.ones(
        (1), dtype=torch.float32
    )
    assert linear_layer.outputs_QParams.scale_factor == torch.ones(
        (1), dtype=torch.float32
    )
    assert linear_layer.outputs_QParams.zero_point == torch.zeros(
        (1), dtype=torch.int32
    )


def simulate_linear_forward(linear_layer, inputs):
    linear_layer.inputs_QParams.update_quant_params(inputs)
    linear_layer.weight_QParams.update_quant_params(linear_layer.weight)
    linear_layer.bias_QParams.update_quant_params(linear_layer.bias)

    inputs = SimQuant.apply(inputs, linear_layer.inputs_QParams)
    weight = SimQuant.apply(linear_layer.weight, linear_layer.weight_QParams)
    bias = SimQuant.apply(linear_layer.bias, linear_layer.bias_QParams)

    outputs = F.linear(inputs, weight, bias)

    min_float = outputs.min()
    max_float = outputs.max()

    min_quant = linear_layer.outputs_QParams.min_quant
    max_quant = linear_layer.outputs_QParams.max_quant
    scale_factor = (max_float - min_float) / (max_quant.float() - min_quant.float())
    scale_factor = torch.tensor([scale_factor.item()], dtype=torch.float32)
    zero_point = max_quant - (max_float / scale_factor)
    zero_point = zero_point.round_().clamp(min_quant, max_quant)
    zero_point = torch.tensor([zero_point.item()], dtype=torch.int32)

    return min_float, max_float, scale_factor, zero_point, outputs


def test_update_quant_params_of_outputs_QParams_in_forward(linear_layer) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    min_float, max_float, scale_factor, zero_point, _ = simulate_linear_forward(
        linear_layer, inputs
    )

    assert linear_layer.outputs_QParams.min_float == min_float
    assert linear_layer.outputs_QParams.max_float == max_float
    assert torch.allclose(
        linear_layer.outputs_QParams.scale_factor,
        scale_factor,
        atol=1e-5,
    )
    assert torch.equal(linear_layer.outputs_QParams.zero_point, zero_point)


def test_forward(linear_layer) -> None:
    expected_outputs = linear_layer.forward(inputs)
    _, _, _, _, tmp_outputs = simulate_linear_forward(linear_layer, inputs)
    linear_layer.outputs_QParams.update_quant_params(tmp_outputs)
    actual_outputs = linear_layer.outputs_QParams.dequantize(
        linear_layer.outputs_QParams.quantize(tmp_outputs)
    )
    assert torch.allclose(
        expected_outputs,
        actual_outputs,
        atol=1e-5,
    )


def test_get_quantized_weights_in_precompute(linear_layer) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

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
    assert torch.equal(linear_layer.q_weights, expected_q_weight)


def test_get_quantized_bias_in_precompute(linear_layer) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    expected_scale_factor = (
        linear_layer.inputs_QParams.scale_factor
        * linear_layer.weight_QParams.scale_factor
    )
    assert torch.allclose(
        linear_layer.bias_QParams.scale_factor,
        expected_scale_factor,
        atol=1e-10,
    )
    assert torch.equal(linear_layer.bias_QParams.zero_point, torch.tensor([0]))
    expected_quant_bits = (linear_layer.inputs_QParams.quant_bits + 1) + (
        linear_layer.weight_QParams.quant_bits + 1
    )
    assert linear_layer.bias_QParams.quant_bits == expected_quant_bits

    expected_q_bias = torch.tensor(
        [9031, -9031, 5419, 3612, -3612, 1806, 7225, -7225, 3612, 10838],
        dtype=torch.int32,
    )
    assert torch.equal(linear_layer.q_bias, expected_q_bias)


def test_get_scale_fator_M_in_precompute(linear_layer) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    input_QParams_scale_factor = torch.tensor([0.0058823530562222], dtype=torch.float32)
    weight_QParams_scale_factor = torch.tensor(
        [0.00941176526248455], dtype=torch.float32
    )
    _, _, output_QParams_scale_factor, _ = simulate_linear_forward(linear_layer, inputs)

    expected_scale_factor_M = (
        input_QParams_scale_factor
        * weight_QParams_scale_factor
        / output_QParams_scale_factor
    )
    assert torch.allclose(
        linear_layer.scale_factor_M, expected_scale_factor_M, atol=1e-3
    )


def test_int_forward(linear_layer) -> None:
    outputs = linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    # Quantize the input before passing to int_forward
    q_input = linear_layer.inputs_QParams.quantize(inputs)
    q_output = linear_layer.int_forward(q_input)

    expected_q_output = linear_layer.outputs_QParams.quantize(outputs)

    assert torch.allclose(q_output, expected_q_output, atol=1)
