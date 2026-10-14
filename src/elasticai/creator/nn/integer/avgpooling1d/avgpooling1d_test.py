import pytest
import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.avgpooling1d import AVGPooling1d


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
    return torch.randn(3, 6, 4)  # (batch, seq_len, channels)


@pytest.fixture
def layer() -> AVGPooling1d:
    return AVGPooling1d(
        name="avgpooling1d_0",
        in_features=6,
        out_features=1,
        in_num_dimensions=4,
        out_num_dimensions=4,
        quant_bits=8,
        device="cpu",
    )


def test_forward_averages_over_seq_len(layer, inputs) -> None:
    outputs = layer(inputs, enable_simquant=False)

    assert torch.allclose(outputs, inputs.mean(dim=1, keepdim=True))


def test_int_forward_matches_forward(layer, inputs) -> None:
    train(layer, inputs)
    layer.precompute()

    q_outputs = layer.int_forward(layer.inputs_QParams.quantize(inputs))

    assert q_outputs.shape == (3, 1, 4)
    assert q_outputs.dtype == torch.int32
    assert max_error_in_steps(layer, q_outputs, layer(inputs)) <= 1


def test_creates_design(layer, inputs) -> None:
    train(layer, inputs)
    layer.precompute()

    names = saved_file_names(layer.create_design("avgpooling1d_0"))

    assert names == {
        "avgpooling1d_0.vhd",
        "avgpooling1d_0_ram.vhd",
        "avgpooling1d_0_tb.vhd",
    }
