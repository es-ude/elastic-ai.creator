import torch

from .linear import Linear


def test_inference_of_multidimensional_data() -> None:
    linear = Linear(
        total_bits=16, frac_bits=8, in_features=3, out_features=2, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data)

    inputs = torch.tensor([1.0, 2.0, 3.0])
    expected = [6.0, 6.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_overflow_behaviour() -> None:
    linear = Linear(
        total_bits=4, frac_bits=1, in_features=2, out_features=1, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data) * 2

    inputs = torch.tensor([2.5, -1.0])
    expected = [3.0]  # quantize(2.5 * 2 - 1.0 * 2)
    actual = linear(inputs).tolist()

    assert expected == actual


def test_underflow_behaviour() -> None:
    linear = Linear(
        total_bits=4, frac_bits=1, in_features=1, out_features=1, bias=False
    )
    linear.weight.data = torch.ones_like(linear.weight.data) * 0.5

    inputs = torch.tensor([0.5])
    expected = [0.0]
    actual = linear(inputs).tolist()

    assert expected == actual


def test_bias_addition() -> None:
    linear = Linear(
        total_bits=16, frac_bits=8, in_features=1, out_features=1, bias=True
    )
    linear.weight.data = torch.ones_like(linear.weight.data)
    linear.bias.data = torch.ones_like(linear.bias.data) * 2

    inputs = torch.tensor([3.0])
    expected = [5.0]
    actual = linear(inputs).tolist()

    assert expected == actual
