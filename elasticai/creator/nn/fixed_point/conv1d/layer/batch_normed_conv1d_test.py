import pytest
import torch

from .batch_normed_conv1d import BatchNormedConv1d


def conv1d(signal_length: int, bias: bool, affine: bool) -> BatchNormedConv1d:
    return BatchNormedConv1d(
        total_bits=4,
        frac_bits=1,
        kernel_size=2,
        signal_length=signal_length,
        bias=bias,
        in_channels=1,
        out_channels=1,
        bn_affine=affine,
        bn_momentum=1,
    )


@pytest.fixture
def batched_input() -> torch.Tensor:
    return torch.rand(3, 15)


def test_that_output_is_flat_for_unbatched_input() -> None:
    conv = conv1d(signal_length=15, bias=False, affine=False)
    input_data = torch.rand(15)
    prediction = conv(input_data)
    tensor_rank = prediction.dim()
    flat_tensor_rank = 1
    assert flat_tensor_rank == tensor_rank


def test_that_batch_dimension_is_kept(batched_input: torch.Tensor) -> None:
    conv = conv1d(signal_length=15, bias=False, affine=False)
    prediction = conv(batched_input)
    batch_dimension = prediction.shape[0]
    expected_batch_dimension = batched_input.shape[0]
    assert expected_batch_dimension == batch_dimension


def test_that_batched_output_is_flat(batched_input: torch.Tensor) -> None:
    conv = conv1d(signal_length=15, bias=False, affine=False)
    prediction = conv(batched_input)
    tensor_rank = prediction.dim()
    expected_rank = 2
    assert expected_rank == tensor_rank
