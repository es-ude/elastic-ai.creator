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
        out_channels=2,
        bn_affine=affine,
        bn_momentum=1,
    )


def test_output_contains_correct_number_of_output_channels() -> None:
    conv = conv1d(signal_length=15, bias=False, affine=False)
    input_data = torch.rand(1, 15)
    prediction = conv(input_data)
    num_channels, _ = prediction.shape
    assert num_channels == 2


def test_that_batch_dimension_is_kept() -> None:
    conv = conv1d(signal_length=15, bias=False, affine=False)
    input_data = torch.rand(3, 1, 15)
    prediction = conv(input_data)
    batch_dimension = prediction.shape[0]
    assert batch_dimension == 3
