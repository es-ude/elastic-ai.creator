import pytest
import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.batchnorm1d import BatchNorm1d


def train(layer, inputs: torch.Tensor, **forward_kwargs) -> None:
    layer.train()
    for _ in range(3):
        layer(inputs, **forward_kwargs)
    layer.eval()


def saved_file_names(design) -> set[str]:
    destination = InMemoryPath("build", parent=None)
    design.save_to(destination)
    names = set()
    stack = [destination]
    while stack:
        for child in stack.pop().children.values():
            if isinstance(child, InMemoryFile):
                names.add(child.name)
            else:
                stack.append(child)
    return names


def max_error_in_steps(layer, q_outputs: torch.Tensor, expected: torch.Tensor) -> float:
    dequantized = layer.outputs_QParams.dequantize(q_outputs)
    step = layer.outputs_QParams.scale_factor
    # Rounded, so that float rounding errors do not count as a step.
    return round(float(((dequantized - expected).abs() / step).max()), 3)


@pytest.fixture
def inputs() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(3, 5, 4)  # (batch, num_dimensions, features)


@pytest.fixture
def layer() -> BatchNorm1d:
    return BatchNorm1d(
        name="batchnorm1d_0",
        norm_dim=4,
        in_features=4,
        out_features=4,
        num_dimensions=5,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        quant_bits=8,
        device="cpu",
    )


def test_forward_normalizes_with_running_statistics_in_eval_mode(layer, inputs) -> None:
    layer.eval()

    outputs = layer(inputs, None, enable_simquant=False)

    expected = (inputs - layer.running_mean) / torch.sqrt(
        layer.running_var + layer.eps
    ) * layer.weight + layer.bias
    assert torch.allclose(outputs, expected)


def test_int_forward_returns_quantized_outputs(layer, inputs) -> None:
    train(layer, inputs, given_inputs_QParams=None)
    layer.precompute()

    q_outputs = layer.int_forward(layer.inputs_QParams.quantize(inputs))

    assert q_outputs.shape == (3, 5, 4)
    assert q_outputs.dtype == torch.int32


def test_creates_design(layer, inputs) -> None:
    train(layer, inputs, given_inputs_QParams=None)
    layer.precompute()

    names = saved_file_names(layer.create_design("batchnorm1d_0"))

    assert names == {
        "batchnorm1d_0.vhd",
        "batchnorm1d_0_w_rom.vhd",
        "batchnorm1d_0_b_rom.vhd",
        "batchnorm1d_0_ram.vhd",
        "batchnorm1d_0_tb.vhd",
    }
