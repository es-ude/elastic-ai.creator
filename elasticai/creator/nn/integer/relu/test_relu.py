import pytest
import torch

from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.relu.relu import ReLU

inputs = torch.tensor([[1.0, -1.0, 0.0], [0.5, -0.5, 2.0]], dtype=torch.float32)


@pytest.fixture
def relu_layer() -> ReLU:
    return ReLU(name="test_relu", quant_bits=8)


def test_relu_layer_initialization_input_quant_bits(relu_layer) -> None:
    assert relu_layer.inputs_QParams.quant_bits == 8


def test_relu_layer_initialization_output_quant_bits(relu_layer) -> None:
    assert relu_layer.outputs_QParams.quant_bits == 8


def test_relu_layer_initialization_input_observer(relu_layer) -> None:
    assert isinstance(relu_layer.inputs_QParams.observer, GlobalMinMaxObserver)


def test_relu_layer_initialization_output_observer(relu_layer) -> None:
    assert isinstance(relu_layer.outputs_QParams.observer, GlobalMinMaxObserver)


def test_not_update_quant_params_of_inputs_QParams_in_forward(relu_layer) -> None:
    relu_layer.eval()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)
    assert relu_layer.inputs_QParams.min_float == torch.ones((1), dtype=torch.float32)
    assert relu_layer.inputs_QParams.max_float == torch.ones((1), dtype=torch.float32)
    assert relu_layer.inputs_QParams.scale_factor == torch.ones(
        (1), dtype=torch.float32
    )
    assert relu_layer.inputs_QParams.zero_point == torch.zeros((1), dtype=torch.int32)


def test_use_given_inputs_QParams_in_forward(relu_layer):
    relu_layer.train()
    given_inputs_QParams = AsymmetricSignedQParams(
        quant_bits=6, observer=GlobalMinMaxObserver()
    )
    relu_layer.forward(inputs, given_inputs_QParams)
    assert relu_layer.inputs_QParams == given_inputs_QParams


def calculate_scale_factor(relu_layer, inputs):
    min_float = inputs.min()
    max_float = inputs.max()

    min_quant = relu_layer.inputs_QParams.min_quant
    max_quant = relu_layer.inputs_QParams.max_quant

    scale_factor = (max_float - min_float) / (max_quant.float() - min_quant.float())
    scale_factor = torch.tensor([scale_factor.item()], dtype=torch.float32)

    zero_point = max_quant - (max_float / scale_factor)
    zero_point = zero_point.round_().clamp(min_quant, max_quant)
    zero_point = torch.tensor([zero_point.item()], dtype=torch.int32)

    return scale_factor, zero_point


def test_update_quant_params_of_inputs_QParams_in_forward(relu_layer) -> None:
    relu_layer.train()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    assert relu_layer.inputs_QParams.min_float == torch.tensor(
        [-1.0], dtype=torch.float32
    )
    assert relu_layer.inputs_QParams.max_float == torch.tensor(
        [2.0], dtype=torch.float32
    )

    scale_factor, zero_point = calculate_scale_factor(relu_layer, inputs)
    assert torch.allclose(
        relu_layer.inputs_QParams.scale_factor,
        scale_factor,
        atol=1e-10,
    )
    assert torch.equal(relu_layer.outputs_QParams.zero_point, zero_point)


def test_output_QParams_equal_to_input_QParams_in_forward(relu_layer) -> None:
    relu_layer.train()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    assert relu_layer.outputs_QParams == relu_layer.inputs_QParams


def simulate_forward(relu_layer, inputs):
    relu_layer.inputs_QParams.update_quant_params(inputs)
    inputs = SimQuant.apply(inputs, relu_layer.inputs_QParams)

    outputs = torch.relu(inputs)
    return outputs


def test_forward(relu_layer) -> None:
    relu_layer.train()
    given_inputs_QParams = None
    expected_outputs = relu_layer.forward(inputs, given_inputs_QParams)

    tmp_outputs = simulate_forward(relu_layer, inputs)
    actual_outputs = SimQuant.apply(tmp_outputs, relu_layer.outputs_QParams)

    assert torch.allclose(actual_outputs, expected_outputs, atol=1e-10)


# def test_forward():
#     relu, input = relu_setup()
#     output = relu.forward(input)

#     assert output is not None
#     assert output.shape == input.shape
#     assert output.dtype == torch.float32

#     expected_output = torch.tensor(
#         [[1.0, 0.0, 0.0], [0.5, 0.0, 2.0]], dtype=torch.float32
#     )
#     assert torch.allclose(output, expected_output, atol=1e-2)  # TODO: check atol


# def test_int_forward():
#     relu, input = relu_setup()
#     relu.inputs_QParams.update_quant_params(input)
#     q_input = relu.inputs_QParams.quantize(input)

#     q_output = relu.int_forward(q_input)

#     expected_output = torch.tensor([[42, -43, -43], [0, -43, 127]], dtype=torch.int32)
#     assert q_output is not None
#     assert q_output.shape == q_input.shape
#     assert q_output.dtype == torch.int32
#     assert torch.equal(q_output, expected_output)
