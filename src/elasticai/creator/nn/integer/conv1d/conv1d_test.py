import pytest
import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.conv1d import Conv1d


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
    return torch.randn(2, 2, 8)  # (batch, in_channels, seq_len)


@pytest.fixture
def layer() -> Conv1d:
    torch.manual_seed(0)
    return Conv1d(
        name="conv1d_0",
        in_channels=2,
        out_channels=3,
        kernel_size=3,
        padding=0,
        seq_len=8,
        quant_bits=8,
        device="cpu",
    )


def test_forward_matches_torch_conv1d(layer, inputs) -> None:
    outputs = layer(inputs, enable_simquant=False)

    expected = torch.nn.functional.conv1d(inputs, layer.weight, layer.bias)
    assert torch.allclose(outputs, expected)


def test_int_forward_matches_forward(layer, inputs) -> None:
    train(layer, inputs)
    layer.precompute()

    q_outputs = layer.int_forward(layer.inputs_QParams.quantize(inputs))

    assert q_outputs.shape == (2, 3, 6)
    assert q_outputs.dtype == torch.int32
    assert max_error_in_steps(layer, q_outputs, layer(inputs)) <= 1


def test_creates_design(layer, inputs) -> None:
    train(layer, inputs)
    layer.precompute()
    # create_design reads the padding length set by int_forward.
    layer.int_forward(layer.inputs_QParams.quantize(inputs))

    names = saved_file_names(layer.create_design("conv1d_0"))

    assert names == {
        "conv1d_0.vhd",
        "conv1d_0_w_rom.vhd",
        "conv1d_0_b_rom.vhd",
        "conv1d_0_ram.vhd",
        "conv1d_0_tb.vhd",
    }
