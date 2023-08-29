import torch

from .layer import Conv1d


def make_conv1d() -> Conv1d:
    return Conv1d(
        total_bits=16,
        frac_bits=8,
        in_channels=3,
        out_channels=4,
        kernel_size=2,
        bias=False,
        signal_length=5,
    )


def test_output_shape_of_unbatched_input() -> None:
    layer = make_conv1d()
    input_data = torch.rand(15)
    prediction = layer(input_data)
    assert prediction.shape == (4 * 4,)


def test_output_shape_of_batched_input() -> None:
    layer = make_conv1d()
    input_data = torch.rand(3, 15)
    prediction = layer(input_data)
    assert prediction.shape == (3, 4 * 4)
