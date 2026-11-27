import pytest
import torch
from torch.nn import functional as F

from elasticai.creator.file_generation.in_memory_path import InMemoryFile, InMemoryPath
from elasticai.creator.nn.integer.conv1d import Conv1d
from elasticai.creator.nn.integer.conv1d.conv1d import Conv1d


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
    training_bias_width = layer.bias_QParams.quant_bits
    training_bias_scale = layer.bias_QParams.scale_factor.clone()
    layer.precompute()

    assert training_bias_width == layer.bias_QParams.quant_bits == layer.accumulator_width
    torch.testing.assert_close(training_bias_scale, layer.bias_QParams.scale_factor)
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




@pytest.mark.parametrize("bias_value", [-0.5, 0.5])
def test_conv1d_clips_bias_to_accumulator_range(bias_value):
    layer = Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=1,
        quant_bits=8,
        padding=0,
        seq_len=0,
        name=""
        
    ).eval()
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(bias_value)
    layer.inputs_QParams.scale_factor.fill_(2.0**-10)
    layer.weight_QParams.scale_factor.fill_(2.0**-10)
    layer.outputs_QParams.scale_factor.fill_(2.0**-7)
    layer.precompute()

    torch.testing.assert_close(
        layer.int_forward(torch.zeros((1, 1, 1), dtype=torch.int32)),
        torch.tensor([[[int(bias_value * 32)]]], dtype=torch.int32),
    )

@pytest.mark.parametrize("kernel_size,output_shift", [(50, 17), (20000, 25)])
@pytest.mark.parametrize("negative_inputs", [False, True])
def test_conv1d_accumulator_uses_int32_arithmetic(
    kernel_size, output_shift, negative_inputs
):
    layer = Conv1d(
        in_channels=2,
        out_channels=1,
        kernel_size=kernel_size,
        quant_bits=8,
        padding=0,
        seq_len=0,
        name=""
    ).eval()
    with torch.no_grad():
        layer.weight.fill_(255)
        if layer.bias is not None:
            layer.bias.zero_()

    # Unit input/weight scales and opposite-end zero points give centered
    # operands of magnitude 255, while all stored codes remain valid int8.
    layer.weight_QParams.zero_point.fill_(-128)
    layer.inputs_QParams.zero_point.fill_(127 if negative_inputs else -128)
    layer.outputs_QParams.scale_factor.fill_(2**output_shift)
    if kernel_size == 20000:
        with pytest.warns(RuntimeWarning, match="Overflow could occur"):
            layer.precompute()
    else:
        layer.precompute()

    inputs = torch.full(
        (1, 2, kernel_size), -128 if negative_inputs else 127, dtype=torch.int32
    )
    accumulator = 2 * kernel_size * 255 * 255 * (-1 if negative_inputs else 1)
    accumulator = (accumulator + 2**31) % 2**32 - 2**31
    expected_code = round(accumulator / (2**output_shift))
    # Requantization keeps the final code in range, exposing int32 wraparound
    # rather than hiding it through output saturation.
    assert -128 < expected_code < 127
    torch.testing.assert_close(
        layer.int_forward(inputs),
        torch.tensor([[[expected_code]]], dtype=torch.int32),
    )
