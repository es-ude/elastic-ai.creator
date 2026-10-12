import pytest
import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryPath
from elasticai.creator.nn.integer.MPQ.linear.linear import Linear
from elasticai.creator.nn.integer.quant_utils import (
    AsymmetricSignedQParams,
    GlobalMinMaxObserver,
)


def create_linear(tmp_path, **kwargs) -> Linear:
    layer = Linear(
        name="linear_0",
        in_features=3,
        out_features=2,
        num_dimensions=1,
        quant_data_dir=str(tmp_path),
        device="cpu",
        **kwargs,
    )
    layer.set_quant_bits_from_config(
        {"linear_0.inputs": 8, "linear_0.weights": 8, "linear_0.outputs": 8}
    )
    return layer


def train_and_precompute(layer: Linear, given_inputs_QParams=None) -> None:
    torch.manual_seed(0)
    layer.train()
    for _ in range(3):
        layer(torch.randn(4, 3), given_inputs_QParams)
    layer.eval()
    layer.precompute()


def test_has_bias_by_default(tmp_path):
    layer = create_linear(tmp_path)

    assert layer.bias is not None
    assert layer.bias_QParams.quant_bits == (8 + 1) + (8 + 1)


@pytest.mark.parametrize("bias", [True, False])
def test_creates_design(tmp_path, bias):
    layer = create_linear(tmp_path, bias=bias)
    train_and_precompute(layer)

    layer.create_design("linear_0").save_to(InMemoryPath("build", parent=None))


def test_without_bias_creates_zero_bias(tmp_path):
    layer = create_linear(tmp_path, bias=False)
    train_and_precompute(layer)

    design = layer.create_design("linear_0")

    assert design._bias == [0, 0]
    assert design._b_data_width == (8 + 1) + (8 + 1)


def test_without_bias_int_forward(tmp_path):
    layer = create_linear(tmp_path, bias=False)
    train_and_precompute(layer)
    q_inputs = layer.inputs_QParams.quantize(torch.randn(4, 3))

    q_outputs = layer.int_forward(q_inputs)

    assert q_outputs.shape == (4, 2)


def test_missing_quant_bits_in_config_raises(tmp_path):
    layer = Linear(name="linear_0", in_features=3, out_features=2, device="cpu")

    with pytest.raises(ValueError, match="linear_0.weights"):
        layer.set_quant_bits_from_config({"linear_0.inputs": 8, "linear_0.outputs": 8})


def test_requantizer_strategy(tmp_path):
    layer = create_linear(tmp_path, MPQ_strategy="requantizer")
    prev_inputs_QParams = AsymmetricSignedQParams(
        quant_bits=6, observer=GlobalMinMaxObserver()
    )
    prev_inputs_QParams.update_quant_params(torch.randn(4, 3))
    train_and_precompute(layer, prev_inputs_QParams)
    q_inputs = prev_inputs_QParams.quantize(torch.randn(4, 3))

    q_outputs = layer.int_forward(q_inputs)

    assert q_outputs.shape == (4, 2)
