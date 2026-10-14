import pytest
import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.maxpooling import MaxPooling1d


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
    return torch.randn(2, 3, 8)  # (batch, channels, seq_len)


@pytest.fixture
def layer() -> MaxPooling1d:
    return MaxPooling1d(
        name="maxpooling_0",
        in_features=8,
        out_features=4,
        in_num_dimensions=3,
        out_num_dimensions=3,
        kernel_size=2,
        quant_bits=8,
        device="cpu",
    )


def test_forward_takes_maximum_of_each_window(layer, inputs) -> None:
    outputs = layer(inputs, enable_simquant=False)

    assert torch.equal(outputs, inputs.view(2, 3, 4, 2).amax(dim=-1))


def test_int_forward_matches_forward(layer, inputs) -> None:
    train(layer, inputs)

    q_outputs = layer.int_forward(layer.inputs_QParams.quantize(inputs))

    assert q_outputs.shape == (2, 3, 4)
    assert q_outputs.dtype == torch.int32
    assert max_error_in_steps(layer, q_outputs, layer(inputs)) <= 1


def test_creates_design(layer, inputs) -> None:
    train(layer, inputs)

    names = saved_file_names(layer.create_design("maxpooling_0"))

    assert names == {"maxpooling_0.vhd", "maxpooling_0_ram.vhd", "maxpooling_0_tb.vhd"}
