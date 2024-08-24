import pytest
import torch
from torch.nn import functional as F

from elasticai.creator.nn.integer.linear.linear import Linear
from elasticai.creator.nn.integer.math_operations.MathOperations import MathOperations
from elasticai.creator.nn.integer.quant_utils.calculate_quant_params import (
    calculate_asymmetric_quant_params,
    calculate_symmetric_quant_params,
)
from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M
from elasticai.creator.nn.integer.quant_utils.SimQuant import SimQuant
from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
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


@pytest.fixture
def inputs():
    return torch.tensor(
        [
            [-1.0, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.0, 0.5],
        ],
        dtype=torch.float32,
    )


@pytest.fixture
def eps():
    return torch.tensor(
        (torch.finfo(torch.float32).eps),
        dtype=torch.float32,
        requires_grad=False,
    )


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
def weights_quant_params(linear_layer, eps):
    weights_scale_factor, weights_zero_point, weights_min_float, weights_max_float = (
        calculate_asymmetric_quant_params(
            min_float=linear_layer.weight.min(),
            max_float=linear_layer.weight.max(),
            min_quant=torch.tensor([-128], dtype=torch.int32),
            max_quant=torch.tensor([127], dtype=torch.int32),
            eps=eps,
        )
    )

    return (
        weights_scale_factor,
        weights_zero_point,
        weights_min_float,
        weights_max_float,
    )


@pytest.fixture
def bias_quant_params(linear_layer, eps):
    bias_scale_factor, bias_zero_point, bias_min_float, bias_max_float = (
        calculate_symmetric_quant_params(
            min_float=linear_layer.bias.min(),
            max_float=linear_layer.bias.max(),
            min_quant=torch.tensor([-127], dtype=torch.int32),
            max_quant=torch.tensor([127], dtype=torch.int32),
            eps=eps,
        )
    )
    return (bias_scale_factor, bias_zero_point, bias_min_float, bias_max_float)


@pytest.fixture
def outputs_quant_params(linear_layer, inputs, eps):
    linear_layer.train()
    linear_layer.inputs_QParams.update_quant_params(inputs)
    linear_layer.weight_QParams.update_quant_params(linear_layer.weight)
    linear_layer.bias_QParams.update_quant_params(linear_layer.bias)

    inputs = SimQuant.apply(inputs, linear_layer.inputs_QParams)
    weight = SimQuant.apply(linear_layer.weight, linear_layer.weight_QParams)
    bias = SimQuant.apply(linear_layer.bias, linear_layer.bias_QParams)

    outputs = F.linear(inputs, weight, bias)

    outputs_scale_factor, outputs_zero_point, outputs_min_float, outputs_max_float = (
        calculate_asymmetric_quant_params(
            min_float=outputs.min(),
            max_float=outputs.max(),
            min_quant=torch.tensor([-128], dtype=torch.int32),
            max_quant=torch.tensor([127], dtype=torch.int32),
            eps=eps,
        )
    )

    return (
        outputs,
        outputs_scale_factor,
        outputs_zero_point,
        outputs_min_float,
        outputs_max_float,
    )


@pytest.fixture
def q_weights_subtracted(linear_layer, weights_quant_params):
    weights_scale_factor, weights_zero_point, _, _ = weights_quant_params

    q_weights = linear_layer.weight / weights_scale_factor + weights_zero_point
    q_weights = q_weights.round_().clamp(min=-128, max=127)
    q_weights = q_weights.clone().detach().to(torch.int32)

    q_weights_subtracted = linear_layer.math_ops.intsub(
        q_weights, weights_zero_point, linear_layer.weight_QParams.quant_bits + 1
    )
    return q_weights_subtracted


@pytest.fixture
def q_bias_subtracted(linear_layer, inputs_quant_params, weights_quant_params):
    inputs_scale_factor, _, _, _ = inputs_quant_params
    weights_scale_factor, _, _, _ = weights_quant_params

    new_bias_scale_factor = inputs_scale_factor * weights_scale_factor
    new_bias_zero_point = torch.tensor([0], dtype=torch.int32)
    new_quant_bits = (linear_layer.inputs_QParams.quant_bits + 1) + (
        linear_layer.weight_QParams.quant_bits + 1
    )

    q_bias = linear_layer.bias / new_bias_scale_factor + new_bias_zero_point
    min_quant = torch.tensor([-(2 ** (new_quant_bits - 1)) + 1], dtype=torch.int32)
    max_quant = torch.tensor([2 ** (new_quant_bits - 1) - 1], dtype=torch.int32)
    q_bias = q_bias.round_().clamp(min=min_quant, max=max_quant)
    q_bias = q_bias.clone().detach().to(torch.int32)

    q_bias_subtracted = q_bias - new_bias_zero_point
    q_bias_subtracted = q_bias_subtracted.clone().detach().to(torch.int32)

    return new_bias_scale_factor, new_bias_zero_point, new_quant_bits, q_bias_subtracted


@pytest.fixture
def scale_fator_M(inputs_quant_params, weights_quant_params, outputs_quant_params):
    inputs_scale_factor, _, _, _ = inputs_quant_params
    weights_scale_factor, _, _, _ = weights_quant_params
    (
        _,
        outputs_scale_factor,
        _,
        _,
        _,
    ) = outputs_quant_params

    scale_factor_M = (inputs_scale_factor * weights_scale_factor) / outputs_scale_factor

    scale_factor_M_q_shift, scale_factor_M_q = scaling_M(scale_factor_M)
    return scale_factor_M, scale_factor_M_q_shift, scale_factor_M_q


@pytest.fixture
def q_inputs(inputs, inputs_quant_params):
    inputs_scale_factor, inputs_zero_point, _, _ = inputs_quant_params
    q_inputs = inputs / inputs_scale_factor + inputs_zero_point
    q_inputs = q_inputs.round_().clamp(min=-128, max=127)
    q_inputs = q_inputs.to(torch.int32)
    return q_inputs


@pytest.fixture
def q_outputs(
    linear_layer,
    q_inputs,
    inputs_quant_params,
    q_weights_subtracted,
    q_bias_subtracted,
    scale_fator_M,
    outputs_quant_params,
):
    _, inputs_zero_point, _, _ = inputs_quant_params
    _, _, new_bias_quant_bits, q_bias_subtracted = q_bias_subtracted
    _, scale_fator_M_q_shift, scale_fator_M_q = scale_fator_M
    _, _, outputs_zero_point, _, _ = outputs_quant_params

    q_inputs = linear_layer.math_ops.intsub(
        q_inputs, inputs_zero_point, linear_layer.inputs_QParams.quant_bits + 1
    )
    tmp = linear_layer.math_ops.intmatmul(
        q_inputs, q_weights_subtracted.t(), new_bias_quant_bits
    )

    tmp = linear_layer.math_ops.intadd(tmp, q_bias_subtracted, new_bias_quant_bits + 1)
    tmp = simulate_bitshifting(tmp, scale_fator_M_q_shift, scale_fator_M_q)
    q_outputs = linear_layer.math_ops.intadd(
        tmp,
        outputs_zero_point,
        linear_layer.outputs_QParams.quant_bits,
    )
    return q_outputs


def test_linear_layer_initialization_weight_quant_bits(linear_layer) -> None:
    expected_quant_bits = 8
    actual_quant_bits = linear_layer.weight_QParams.quant_bits
    assert expected_quant_bits == actual_quant_bits


def test_linear_layer_initialization_bias_quant_bits(linear_layer) -> None:
    expected_quant_bits = 8
    actual_quant_bits = linear_layer.bias_QParams.quant_bits
    assert expected_quant_bits == actual_quant_bits


def test_linear_layer_initialization_input_quant_bits(linear_layer) -> None:
    excepted_quant_bits = 8
    actual_quant_bits = linear_layer.inputs_QParams.quant_bits
    assert excepted_quant_bits == actual_quant_bits


def test_linear_layer_initialization_output_quant_bits(linear_layer) -> None:
    excepted_quant_bits = 8
    actual_quant_bits = linear_layer.outputs_QParams.quant_bits
    assert excepted_quant_bits == actual_quant_bits


def test_linear_layer_initialization_weight_observer(linear_layer) -> None:
    expected_observer = GlobalMinMaxObserver
    actual_observer = linear_layer.weight_QParams.observer
    assert isinstance(actual_observer, expected_observer)


def test_linear_layer_initialization_bias_observer(linear_layer) -> None:
    expected_observer = GlobalMinMaxObserver
    actual_observer = linear_layer.bias_QParams.observer
    assert isinstance(actual_observer, expected_observer)


def test_linear_layer_initialization_input_observer(linear_layer) -> None:
    expected_observer = GlobalMinMaxObserver
    actual_observer = linear_layer.inputs_QParams.observer
    assert isinstance(actual_observer, expected_observer)


def test_linear_layer_initialization_output_observer(linear_layer) -> None:
    expected_observer = GlobalMinMaxObserver
    actual_observer = linear_layer.outputs_QParams.observer
    assert isinstance(actual_observer, expected_observer)


def test_linear_layer_initialization_math_ops(linear_layer) -> None:
    expected_math_ops = MathOperations
    actual_math_ops = linear_layer.math_ops
    assert isinstance(actual_math_ops, expected_math_ops)


def test_not_update_quant_params_of_inputs_QParams_in_forward(
    linear_layer, inputs
) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    actual_min_float = linear_layer.inputs_QParams.min_float
    actual_max_float = linear_layer.inputs_QParams.max_float
    actual_scale_factor = linear_layer.inputs_QParams.scale_factor
    actual_zero_point = linear_layer.inputs_QParams.zero_point

    expected_min_float = torch.ones((1), dtype=torch.float32)
    expected_max_float = torch.ones((1), dtype=torch.float32)
    expected_scale_factor = torch.ones((1), dtype=torch.float32)
    expected_zero_point = torch.zeros((1), dtype=torch.int32)

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert expected_scale_factor == actual_scale_factor
    assert expected_zero_point == actual_zero_point


def test_use_given_inputs_QParams_in_forward(linear_layer, inputs) -> None:
    given_inputs_QParams = AsymmetricSignedQParams(
        quant_bits=6, observer=GlobalMinMaxObserver()
    )
    linear_layer.train()
    linear_layer.forward(inputs, given_inputs_QParams)

    expected_inputs_QParams = given_inputs_QParams
    actual_inputs_QParams = linear_layer.inputs_QParams
    assert expected_inputs_QParams == actual_inputs_QParams


def test_update_quant_params_of_inputs_QParams_in_forward(
    linear_layer, inputs, inputs_quant_params
) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    inputs_scale_factor, inputs_zero_point, inputs_min_float, inputs_max_float = (
        inputs_quant_params
    )
    expected_min_float = inputs_min_float
    expected_max_float = inputs_max_float
    expected_scale_factor = inputs_scale_factor
    expected_zero_point = inputs_zero_point

    actual_min_float = linear_layer.inputs_QParams.min_float
    actual_max_float = linear_layer.inputs_QParams.max_float
    actual_scale_factor = linear_layer.inputs_QParams.scale_factor
    actual_zero_point = linear_layer.inputs_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_not_update_quant_params_of_weight_QParams_in_forward(
    linear_layer, inputs
) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    expected_min_float = torch.ones((1), dtype=torch.float32)
    expected_max_float = torch.ones((1), dtype=torch.float32)
    expected_scale_factor = torch.ones((1), dtype=torch.float32)
    expected_zero_point = torch.zeros((1), dtype=torch.int32)

    actual_min_float = linear_layer.weight_QParams.min_float
    actual_max_float = linear_layer.weight_QParams.max_float
    actual_scale_factor = linear_layer.weight_QParams.scale_factor
    actual_zero_point = linear_layer.weight_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_update_quant_params_of_weight_QParams_in_forward(
    linear_layer, inputs, weights_quant_params
) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    weights_scale_factor, weights_zero_point, weights_min_float, weights_max_float = (
        weights_quant_params
    )
    expected_min_float = weights_min_float
    expected_max_float = weights_max_float
    expected_scale_factor = weights_scale_factor
    expected_zero_point = weights_zero_point

    actual_min_float = linear_layer.weight_QParams.min_float
    actual_max_float = linear_layer.weight_QParams.max_float
    actual_scale_factor = linear_layer.weight_QParams.scale_factor
    actual_zero_point = linear_layer.weight_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_not_update_quant_params_of_bias_QParams_in_forward(
    linear_layer, inputs
) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    expected_min_float = torch.ones((1), dtype=torch.float32)
    expected_max_float = torch.ones((1), dtype=torch.float32)
    expected_scale_factor = torch.ones((1), dtype=torch.float32)
    expected_zero_point = torch.zeros((1), dtype=torch.int32)

    actual_min_float = linear_layer.bias_QParams.min_float
    actual_max_float = linear_layer.bias_QParams.max_float
    actual_scale_factor = linear_layer.bias_QParams.scale_factor
    actual_zero_point = linear_layer.bias_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_update_quant_params_of_bias_QParams_in_forward(
    linear_layer, inputs, bias_quant_params
) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    bias_scale_factor, bias_zero_point, bias_min_float, bias_max_float = (
        bias_quant_params
    )
    expected_min_float = bias_min_float
    expected_max_float = bias_max_float
    expected_scale_factor = bias_scale_factor
    expected_zero_point = bias_zero_point

    actual_min_float = linear_layer.bias_QParams.min_float
    actual_max_float = linear_layer.bias_QParams.max_float
    actual_scale_factor = linear_layer.bias_QParams.scale_factor
    actual_zero_point = linear_layer.bias_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_not_update_quant_params_of_outputs_QParams_in_forward(
    linear_layer, inputs
) -> None:
    linear_layer.eval()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    expected_min_float = torch.ones((1), dtype=torch.float32)
    expected_max_float = torch.ones((1), dtype=torch.float32)
    expected_scale_factor = torch.ones((1), dtype=torch.float32)
    expected_zero_point = torch.zeros((1), dtype=torch.int32)

    actual_min_float = linear_layer.outputs_QParams.min_float
    actual_max_float = linear_layer.outputs_QParams.max_float
    actual_scale_factor = linear_layer.outputs_QParams.scale_factor
    actual_zero_point = linear_layer.outputs_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_update_quant_params_of_outputs_QParams_in_forward(
    linear_layer, inputs, outputs_quant_params
) -> None:
    linear_layer.train()
    given_inputs_QParams = None
    linear_layer.forward(inputs, given_inputs_QParams)

    (
        _,
        outputs_scale_factor,
        outputs_zero_point,
        outputs_min_float,
        outputs_max_float,
    ) = outputs_quant_params

    expected_min_float = outputs_min_float
    expected_max_float = outputs_max_float
    expected_scale_factor = outputs_scale_factor
    expected_zero_point = outputs_zero_point

    actual_min_float = linear_layer.outputs_QParams.min_float
    actual_max_float = linear_layer.outputs_QParams.max_float
    actual_scale_factor = linear_layer.outputs_QParams.scale_factor
    actual_zero_point = linear_layer.outputs_QParams.zero_point

    assert expected_min_float == actual_min_float
    assert expected_max_float == actual_max_float
    assert torch.allclose(expected_scale_factor, actual_scale_factor, atol=1e-5)
    assert torch.equal(expected_zero_point, actual_zero_point)


def test_forward(linear_layer, inputs, outputs_quant_params) -> None:
    expected_outputs = linear_layer.forward(inputs)
    (
        tmp_outputs,
        _,
        _,
        _,
        _,
    ) = outputs_quant_params
    linear_layer.outputs_QParams.update_quant_params(tmp_outputs)
    actual_outputs = SimQuant.apply(tmp_outputs, linear_layer.outputs_QParams)
    assert torch.allclose(
        expected_outputs,
        actual_outputs,
        atol=1e-5,
    )


def test_get_quantized_weights_in_precompute(
    linear_layer, inputs, q_weights_subtracted
) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    expected_q_weight = q_weights_subtracted
    actual_q_weights = linear_layer.q_weights
    assert torch.equal(actual_q_weights, expected_q_weight)


def test_get_quantized_bias_in_precompute(
    linear_layer, inputs, q_bias_subtracted
) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    (
        expected_new_bias_scale_factor,
        expected_new_bias_zero_point,
        expected_new_quant_bits,
        expected_q_bias,
    ) = q_bias_subtracted

    actual_new_bias_scale_factor = linear_layer.bias_QParams.scale_factor
    actual_new_zero_point = linear_layer.bias_QParams.zero_point
    actual_new_quant_bits = linear_layer.bias_QParams.quant_bits
    actual_q_bias = linear_layer.q_bias

    assert torch.allclose(
        actual_new_bias_scale_factor,
        expected_new_bias_scale_factor,
        atol=1e-10,
    )
    assert torch.equal(actual_new_zero_point, expected_new_bias_zero_point)
    assert actual_new_quant_bits == expected_new_quant_bits
    assert torch.equal(actual_q_bias, expected_q_bias)


def test_get_scale_fator_M_in_precompute(linear_layer, inputs, scale_fator_M) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    (
        expected_scale_factor_M,
        expected_scale_factor_M_q_shift,
        expected_scale_factor_M_q,
    ) = scale_fator_M
    actual_scale_factor_M = linear_layer.scale_factor_M
    actual_scale_factor_M_q = linear_layer.scale_factor_m_q
    actual_scale_factor_M_q_shift = linear_layer.scale_factor_m_q_shift
    assert expected_scale_factor_M == actual_scale_factor_M
    assert expected_scale_factor_M_q == actual_scale_factor_M_q
    assert expected_scale_factor_M_q_shift == actual_scale_factor_M_q_shift


def test_int_forward(linear_layer, inputs, q_inputs, q_outputs) -> None:
    linear_layer.forward(inputs)
    linear_layer.eval()
    linear_layer.precompute()

    assert linear_layer.precomputed == True
    expected_q_output = q_outputs
    actual_q_output = linear_layer.int_forward(q_inputs)
    print("expected_q_output", expected_q_output)
    print("actual_q_output", actual_q_output)
    assert torch.allclose(actual_q_output, expected_q_output, atol=1)
