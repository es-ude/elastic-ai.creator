import torch

from elasticai.creator.nn.conv1d.layer import FPConv1d


def make_conv1d() -> FPConv1d:
    return FPConv1d(
        total_bits=16,
        frac_bits=8,
        in_channels=3,
        out_channels=4,
        kernel_size=2,
        bias=False,
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
