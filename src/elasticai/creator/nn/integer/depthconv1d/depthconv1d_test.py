import pytest
import torch

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.depthconv1d import DepthConv1d


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
    return torch.randn(2, 3, 8)  # (batch, in_channels, seq_len)


@pytest.fixture
def layer() -> DepthConv1d:
    torch.manual_seed(0)
    return DepthConv1d(
        name="depthconv1d_0",
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=0,
        seq_len=8,
        quant_bits=8,
        device="cpu",
    )


def test_convolves_each_channel_separately(layer) -> None:
    assert layer.groups == 3
    assert layer.weight.shape == (3, 1, 3)


def test_forward_matches_torch_conv1d(layer, inputs) -> None:
    outputs = layer(inputs, enable_simquant=False)

    expected = torch.nn.functional.conv1d(inputs, layer.weight, layer.bias, groups=3)
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

    names = saved_file_names(layer.create_design("depthconv1d_0"))

    assert names == {
        "depthconv1d_0.vhd",
        "depthconv1d_0_w_rom.vhd",
        "depthconv1d_0_b_rom.vhd",
        "depthconv1d_0_ram.vhd",
        "depthconv1d_0_tb.vhd",
    }


def create_padded_layer(padding: int | str) -> DepthConv1d:
    torch.manual_seed(0)
    return DepthConv1d(
        name="depthconv1d_0",
        in_channels=3,
        out_channels=3,
        kernel_size=3,
        padding=padding,
        seq_len=8,
        quant_bits=8,
        device="cpu",
    )


@pytest.mark.parametrize("padding", [1, "same"])
def test_forward_pads_like_torch_conv1d(inputs, padding) -> None:
    layer = create_padded_layer(padding)

    outputs = layer(inputs, enable_simquant=False)

    expected = torch.nn.functional.conv1d(
        inputs, layer.weight, layer.bias, padding=padding, groups=3
    )
    assert torch.allclose(outputs, expected)


@pytest.mark.parametrize("padding, out_seq_len", [(1, 8), (2, 10), ("same", 8)])
def test_int_forward_with_padding_matches_forward(inputs, padding, out_seq_len) -> None:
    layer = create_padded_layer(padding)
    train(layer, inputs)
    layer.precompute()

    q_outputs = layer.int_forward(layer.inputs_QParams.quantize(inputs))

    assert q_outputs.shape == (2, 3, out_seq_len)
    assert max_error_in_steps(layer, q_outputs, layer(inputs)) <= 1


def test_creates_zero_padding_design(inputs) -> None:
    layer = create_padded_layer(1)
    train(layer, inputs)
    layer.precompute()

    design = layer.create_design("depthconv1d_0")

    assert design._padding_len == 1
    assert "depthconv1d_0.vhd" in saved_file_names(design)
