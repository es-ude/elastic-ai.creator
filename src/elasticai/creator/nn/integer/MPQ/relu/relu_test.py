import pytest
import torch
import torch.nn.functional as F

from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
)
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.relu.relu import ReLU


@pytest.fixture
def relu_layer() -> ReLU:
    return ReLU(name="test_relu", quant_bits=8)


@pytest.fixture
def eps():
    return torch.tensor(
        (torch.finfo(torch.float32).eps),
        dtype=torch.float32,
        requires_grad=False,
    )


@pytest.fixture
def inputs() -> torch.FloatTensor:
    return torch.tensor([[1.0, -1.0, 0.0], [0.5, -0.5, 2.0]], dtype=torch.float32)


@pytest.fixture
def inputs_quant_params(inputs, eps):
    inputs_scale_factor, inputs_zero_point, inputs_min_float, inputs_max_float = (
        calculate_asymmetric_quant_params(
            min_float=inputs.min(),
            max_float=inputs.max(),
            min_quant=torch.tensor([-128], dtype=torch.int32),
            max_quant=torch.tensor([127], dtype=torch.int32),
            eps=eps,
        )
    )
    return (inputs_scale_factor, inputs_zero_point, inputs_min_float, inputs_max_float)


@pytest.fixture
def outputs_quant_params(inputs, relu_layer, inputs_quant_params):
    relu_layer.train()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    outputs_scale_factor, outputs_zero_point, outputs_min_float, outputs_max_float = (
        inputs_quant_params
    )

    inputs = SimQuant.apply(inputs, relu_layer.inputs_QParams)
    outputs = F.relu(inputs).to(torch.float32)
    return (
        outputs,
        outputs_scale_factor,
        outputs_zero_point,
        outputs_min_float,
        outputs_max_float,
    )


@pytest.fixture
def q_inputs(inputs, inputs_quant_params):
    inputs_scale_factor, inputs_zero_point, _, _ = inputs_quant_params
    q_inputs = inputs / inputs_scale_factor + inputs_zero_point
    q_inputs = q_inputs.round_().clamp(min=-128, max=127)
    q_inputs = q_inputs.to(torch.int32)
    return q_inputs


@pytest.fixture
def q_outputs(q_inputs, inputs_quant_params):
    _, inputs_zero_point, _, _ = inputs_quant_params
    q_outputs = torch.maximum(q_inputs, inputs_zero_point.clone().detach())
    return q_outputs


def test_relu_layer_initialization_input_quant_bits(relu_layer) -> None:
    expected_quant_bits = 8
    actual_quant_bits = relu_layer.inputs_QParams.quant_bits
    assert expected_quant_bits == actual_quant_bits


def test_relu_layer_initialization_output_quant_bits(relu_layer) -> None:
    expected_quant_bits = 8
    actual_quant_bits = relu_layer.outputs_QParams.quant_bits
    assert expected_quant_bits == actual_quant_bits

    assert relu_layer.outputs_QParams.quant_bits == 8


def test_relu_layer_initialization_input_observer(relu_layer) -> None:
    expected_observer = GlobalMinMaxObserver
    actual_observer = relu_layer.inputs_QParams.observer
    assert isinstance(actual_observer, expected_observer)


def test_relu_layer_initialization_output_observer(relu_layer) -> None:
    expected_observer = GlobalMinMaxObserver
    actual_observer = relu_layer.outputs_QParams.observer
    assert isinstance(actual_observer, expected_observer)


def test_not_update_quant_params_of_inputs_QParams_in_forward(
    relu_layer, inputs
) -> None:
    relu_layer.eval()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    actual_min_float = relu_layer.inputs_QParams.min_float
    actual_max_float = relu_layer.inputs_QParams.max_float
    actual_scale_factor = relu_layer.inputs_QParams.scale_factor
    actual_zero_point = relu_layer.inputs_QParams.zero_point

    expected_min_float = torch.ones((1), dtype=torch.float32)
    expected_max_float = torch.ones((1), dtype=torch.float32)
    expected_scale_factor = torch.ones((1), dtype=torch.float32)
    expected_zero_point = torch.zeros((1), dtype=torch.int32)

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert expected_scale_factor == actual_scale_factor
    assert expected_zero_point == actual_zero_point


def test_use_given_inputs_QParams_in_forward(relu_layer, inputs):
    given_inputs_QParams = AsymmetricSignedQParams(
        quant_bits=6, observer=GlobalMinMaxObserver()
    )
    relu_layer.train()
    relu_layer.forward(inputs, given_inputs_QParams)

    expected_inputs_QParams = given_inputs_QParams
    actual_inputs_QParams = relu_layer.inputs_QParams
    assert expected_inputs_QParams == actual_inputs_QParams


def test_update_quant_params_of_inputs_QParams_in_forward(
    relu_layer, inputs, inputs_quant_params
) -> None:
    relu_layer.train()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    inputs_scale_factor, inputs_zero_point, inputs_min_float, inputs_max_float = (
        inputs_quant_params
    )
    expected_min_float = inputs_min_float
    expected_max_float = inputs_max_float
    expected_scale_factor = inputs_scale_factor
    expected_zero_point = inputs_zero_point

    actual_min_float = relu_layer.inputs_QParams.min_float
    actual_max_float = relu_layer.inputs_QParams.max_float
    actual_scale_factor = relu_layer.inputs_QParams.scale_factor
    actual_zero_point = relu_layer.inputs_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_output_QParams_equal_to_input_QParams_in_forward(
    relu_layer, inputs, outputs_quant_params
) -> None:
    relu_layer.train()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    (
        _,
        expected_outputs_scale_factor,
        expected_outputs_zero_point,
        expected_outputs_min_float,
        expected_outputs_max_float,
    ) = outputs_quant_params

    actual_outputs_scale_factor = relu_layer.outputs_QParams.scale_factor
    actual_outputs_zero_point = relu_layer.outputs_QParams.zero_point

    actual_outputs_min_float = relu_layer.outputs_QParams.min_float
    actual_outputs_max_float = relu_layer.outputs_QParams.max_float

    assert torch.allclose(
        expected_outputs_scale_factor, actual_outputs_scale_factor, atol=1e-5
    )
    assert expected_outputs_zero_point == actual_outputs_zero_point
    assert expected_outputs_min_float == actual_outputs_min_float
    assert expected_outputs_max_float == actual_outputs_max_float


def test_forward(relu_layer, inputs, outputs_quant_params) -> None:
    relu_layer.train()
    given_inputs_QParams = None
    actual_outputs = relu_layer.forward(inputs, given_inputs_QParams)

    outputs, _, _, _, _ = outputs_quant_params
    expected_outputs = SimQuant.apply(outputs, relu_layer.outputs_QParams)
    assert torch.allclose(actual_outputs, expected_outputs, atol=1e-10)


def test_int_forward(relu_layer, inputs, q_inputs, q_outputs):
    relu_layer.train()
    given_inputs_QParams = None
    relu_layer.forward(inputs, given_inputs_QParams)

    relu_layer.eval()
    actual_q_outputs = relu_layer.int_forward(q_inputs)
    expected_q_outputs = q_outputs
    assert torch.equal(actual_q_outputs, expected_q_outputs)
