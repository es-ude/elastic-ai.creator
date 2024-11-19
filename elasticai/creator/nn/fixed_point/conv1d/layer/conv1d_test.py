import pytest
import torch

from .conv1d import Conv1d


@pytest.fixture
def conv1d() -> Conv1d:
    return Conv1d(
        total_bits=16,
        frac_bits=8,
        in_channels=3,
        out_channels=4,
        kernel_size=2,
        bias=False,
        signal_length=5,
    )


@pytest.fixture
def build_5_total_1_frac_conv1d():
    def build(in_channels, out_channels, kernel_size, signal_length):
        return Conv1d(
            total_bits=5,
            frac_bits=1,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            signal_length=signal_length,
        )

    return build


@pytest.fixture
def to_1d_input_tensor():
    def get_list_nesting_depth(l):
        level = 0
        while isinstance(l, list):
            level += 1
            if len(l) > 0:
                l = l[0]
            else:
                break
        return level

    def convert(data: list):
        correctly_nested = data
        actual_depth = get_list_nesting_depth(data)
        required_depth = 3
        for _ in range(required_depth - actual_depth):
            correctly_nested = [correctly_nested]
        return torch.tensor(correctly_nested)

    return convert


@pytest.fixture
def conv_with_4_total_1_frac_and_kernel_size_2() -> Conv1d:
    return Conv1d(
        total_bits=4,
        frac_bits=1,
        kernel_size=2,
        signal_length=2,
        bias=False,
        in_channels=1,
        out_channels=1,
    )


@pytest.fixture
def tensor_of_size_3_15() -> torch.Tensor:
    return torch.rand(3, 15)


@pytest.fixture
def batched_input(tensor_of_size_3_15):
    return tensor_of_size_3_15


@pytest.fixture
def tensor_of_size_3_5_7() -> torch.Tensor:
    return torch.rand(3, 5, 7)


def test_that_batch_dimension_is_kept(
    build_5_total_1_frac_conv1d, batched_input: torch.Tensor
) -> None:
    in_channels = 5
    signal_length = 7
    batch_size = 3
    kernel_size = 2
    out_channels = 9
    input_data = torch.rand(batch_size, in_channels, signal_length)
    conv1d = build_5_total_1_frac_conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        signal_length=signal_length,
    )
    prediction = conv1d(input_data)
    expected_size = (batch_size, out_channels, signal_length - kernel_size + 1)

    assert expected_size == prediction.size()


def test_that_batched_output_is_flat(
    conv1d: Conv1d, batched_input: torch.Tensor
) -> None:
    prediction = conv1d(batched_input)
    tensor_rank = prediction.dim()
    expected_rank = 2
    assert expected_rank == tensor_rank


@pytest.mark.parametrize(
    "signal_length, inputs, expected",
    [
        (2, [1.0, 1.0], [2.0]),
        (3, [1.0, 2.0, 4.0], [3.0, 6.0]),
    ],
)
def test_ones_weight_kernel_convolution(
    signal_length: int, inputs: list[float], expected: list[float], to_1d_input_tensor
) -> None:
    input_tensor = to_1d_input_tensor(inputs)
    conv = Conv1d(
        total_bits=16,
        frac_bits=8,
        in_channels=1,
        out_channels=1,
        bias=False,
        signal_length=signal_length,
        kernel_size=2,
    )
    conv.weight.data = torch.ones_like(conv.weight)
    prediction = conv(input_tensor)
    expected = to_1d_input_tensor(expected).tolist()
    assert expected == prediction.tolist()


def test_no_overflow_after_multiplication(
    conv_with_4_total_1_frac_and_kernel_size_2, to_1d_input_tensor
) -> None:
    conv = conv_with_4_total_1_frac_and_kernel_size_2
    conv.weight.data = 3.5 * torch.ones_like(conv.weight)
    prediction = conv(to_1d_input_tensor([3.5, -3]))
    expected = to_1d_input_tensor(
        1.5
    ).tolist()  # quantize(3.5 * 3.5 - 3 * 3.5) == quantize(1.75) == 1.5
    assert expected == prediction.tolist()


def test_no_underflow_after_multiplication(
    conv_with_4_total_1_frac_and_kernel_size_2, to_1d_input_tensor
) -> None:
    conv = conv_with_4_total_1_frac_and_kernel_size_2
    conv.weight.data = 0.5 * torch.ones_like(conv.weight)
    prediction = conv(to_1d_input_tensor([0.5, 0]))
    expected = to_1d_input_tensor(
        0
    ).tolist()  # quantize(0.5 * 0.5 + 0.5 * 0) == quantize(0.25) == 0
    assert expected == prediction.tolist()


@pytest.mark.parametrize(
    "inputs, expected",
    [([2.0, 2.0], [3.5]), ([-2.0, -3.0], [-4.0])],
)
def test_fxp_operations_additive_over_and_underflow(
    conv_with_4_total_1_frac_and_kernel_size_2,
    inputs: list[float],
    expected: list[float],
    to_1d_input_tensor,
) -> None:
    conv = conv_with_4_total_1_frac_and_kernel_size_2
    conv.weight.data = torch.ones_like(conv.weight)
    input_tensor = torch.tensor(to_1d_input_tensor(inputs))
    prediction = conv(input_tensor)
    expected = to_1d_input_tensor(expected).tolist()
    assert expected == prediction.tolist()


def test_ensure_bias_is_0_5(to_1d_input_tensor) -> None:
    conv = Conv1d(
        total_bits=4,
        frac_bits=1,
        kernel_size=2,
        signal_length=2,
        bias=True,
        in_channels=1,
        out_channels=1,
    )
    conv.weight.data = torch.ones_like(conv.weight)
    conv.bias.data = 0.5 * torch.ones_like(conv.bias)  # type: ignore
    inputs = to_1d_input_tensor([0.0, 0.0])
    predictions = conv(inputs).tolist()
    expected = to_1d_input_tensor(0.5).tolist()
    assert expected == predictions
