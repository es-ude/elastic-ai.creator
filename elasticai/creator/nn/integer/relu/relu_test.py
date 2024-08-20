import torch

from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import AsymmetricSignedQParams
from elasticai.creator.nn.integer.relu.relu import ReLU


def relu_setup():
    relu = ReLU(name="test_relu", quant_bits=8)
    input = torch.tensor([[1.0, -1.0, 0.0], [0.5, -0.5, 2.0]], dtype=torch.float32)
    return relu, input


def test_initialization():
    relu, _ = relu_setup()
    assert relu.name == "test_relu"
    assert relu.quant_bits == 8
    assert isinstance(relu.input_QParams, AsymmetricSignedQParams)
    assert isinstance(relu.output_QParams, AsymmetricSignedQParams)
    assert isinstance(relu.input_QParams.observer, GlobalMinMaxObserver)
    assert isinstance(relu.output_QParams.observer, GlobalMinMaxObserver)


def test_forward():
    relu, input = relu_setup()
    output = relu.forward(input)

    assert output is not None
    assert output.shape == input.shape
    assert output.dtype == torch.float32

    expected_output = torch.tensor(
        [[1.0, 0.0, 0.0], [0.5, 0.0, 2.0]], dtype=torch.float32
    )
    assert torch.allclose(output, expected_output, atol=1e-2)  # TODO: check atol


def test_int_forward():
    relu, input = relu_setup()
    relu.input_QParams.update_quant_params(input)
    q_input = relu.input_QParams.quantize(input)

    q_output = relu.int_forward(q_input)

    expected_output = torch.tensor([[42, -43, -43], [0, -43, 127]], dtype=torch.int32)
    assert q_output is not None
    assert q_output.shape == q_input.shape
    assert q_output.dtype == torch.int32
    assert torch.equal(q_output, expected_output)
