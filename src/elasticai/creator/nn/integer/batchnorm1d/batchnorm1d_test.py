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


def train_with_gradients(layer, inputs: torch.Tensor) -> None:
    layer.train()
    for _ in range(3):
        layer(inputs.clone().requires_grad_(), None).sum().backward()
    layer.eval()


def test_running_statistics_are_not_attached_to_the_graph(layer, inputs) -> None:
    train_with_gradients(layer, inputs)

    assert layer.running_mean.grad_fn is None
    assert layer.running_var.grad_fn is None


def test_running_statistics_stay_buffers(layer, inputs) -> None:
    running_mean = layer.running_mean

    train_with_gradients(layer, inputs)

    assert layer.running_mean is running_mean
    assert "running_mean" in dict(layer.named_buffers())


def test_running_statistics_follow_momentum(layer, inputs) -> None:
    layer.train()
    layer(inputs, None, enable_simquant=False)

    expected_mean = 0.9 * torch.zeros(4) + 0.1 * inputs.mean(dim=(0, 1))
    expected_var = 0.9 * torch.ones(4) + 0.1 * inputs.var(dim=(0, 1))
    assert torch.allclose(layer.running_mean, expected_mean)
    assert torch.allclose(layer.running_var, expected_var)


def test_backward_works_in_eval_mode_after_training(layer, inputs) -> None:
    train_with_gradients(layer, inputs)
    eval_inputs = inputs.clone().requires_grad_()

    layer(eval_inputs, None).sum().backward()

    assert eval_inputs.grad is not None


def test_precompute_does_not_change_forward(layer, inputs) -> None:
    train(layer, inputs, given_inputs_QParams=None)
    with torch.no_grad():
        before = layer(inputs, None)
        layer.precompute()
        after = layer(inputs, None)

    assert torch.equal(before, after)


def test_precompute_keeps_bias_qparams(layer, inputs) -> None:
    train(layer, inputs, given_inputs_QParams=None)
    names = ["scale_factor", "zero_point", "min_quant", "max_quant"]
    before = {name: getattr(layer.bias_QParams, name).clone() for name in names}

    layer.precompute()

    for name in names:
        assert torch.equal(getattr(layer.bias_QParams, name), before[name]), name


def test_precompute_twice_gives_same_quantized_bias(layer, inputs) -> None:
    train(layer, inputs, given_inputs_QParams=None)
    layer.precompute()
    q_bias = layer.q_bias.clone()

    layer.precompute()

    assert torch.equal(layer.q_bias, q_bias)


def test_precompute_does_not_change_state_dict_keys(layer, inputs) -> None:
    train(layer, inputs, given_inputs_QParams=None)
    keys = set(layer.state_dict())

    layer.precompute()

    # Otherwise a state_dict saved after precompute does not load into a new
    # layer.
    assert set(layer.state_dict()) == keys
