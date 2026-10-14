import pytest
import torch

from elasticai.creator.nn.integer.layernorm1d import LayerNorm1d


def create_layernorm() -> LayerNorm1d:
    return LayerNorm1d(name="layernorm", norm_dim=8, quant_bits=8, device="cpu")


def train(layer: LayerNorm1d, inputs: torch.Tensor) -> None:
    layer.train()
    for _ in range(3):
        layer(inputs)
    layer.eval()


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(4, 8) * 3 + 1


def test_forward_matches_torch_layernorm_without_simquant(inputs) -> None:
    layer = create_layernorm()
    torch.nn.init.uniform_(layer.weight, -1, 1)
    torch.nn.init.uniform_(layer.bias, -1, 1)

    outputs = layer(inputs, enable_simquant=False)

    expected = torch.nn.functional.layer_norm(
        inputs, (8,), layer.weight, layer.bias, layer.eps
    )
    assert torch.allclose(outputs, expected, atol=1e-6)


def test_int_forward_matches_forward(inputs) -> None:
    layer = create_layernorm()
    torch.nn.init.uniform_(layer.weight, -1, 1)
    torch.nn.init.uniform_(layer.bias, -1, 1)
    train(layer, inputs)
    layer.precompute()

    q_outputs = layer.int_forward(layer.inputs_QParams.quantize(inputs))

    outputs = layer.outputs_QParams.dequantize(q_outputs)
    expected = layer(inputs)
    step = layer.outputs_QParams.scale_factor
    assert torch.all((outputs - expected).abs() <= 2 * step)
