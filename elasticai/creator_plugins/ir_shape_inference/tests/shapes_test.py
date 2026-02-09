import pytest
import torch
import torch.nn as nn

from elasticai.creator_plugins.ir_shape_inference.shapes_calculation_functions import (
    adaptiveavgpool2d_output_shape,
    add_output_shape,
    batchnorm1d_output_shape,
    batchnorm2d_output_shape,
    conv1d_output_shape,
    conv2d_output_shape,
    flatten_output_shape,
    linear_output_shape,
    maxpool1d_output_shape,
    maxpool2d_output_shape,
    relu_output_shape,
    sigmoid_output_shape,
)


def ref_shape(module: nn.Module, x: torch.Tensor) -> tuple:
    module.eval()
    with torch.no_grad():
        return tuple(module(x).shape)


# ---------------------------------------------------------------------------
# linear
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "N, in_features, out_features",
    [
        (4, 10, 5),
        (1, 128, 64),
        (8, 1, 1),
    ],
)
def test_linear(N, in_features, out_features):
    x = torch.zeros(N, in_features)
    expected = ref_shape(nn.Linear(in_features, out_features), x)
    assert linear_output_shape(N, out_features) == expected


# ---------------------------------------------------------------------------
# conv1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape, out_channels, kernel_size, stride, padding, dilation",
    [
        ((2, 3, 16), 8, 3, 1, 0, 1),
        ((2, 3, 16), 8, 3, 2, 1, 1),
        ((2, 3, 32), 4, 5, 1, "same", 1),
        ((2, 3, 20), 6, 3, 1, "valid", 1),
        ((2, 3, 32), 8, 3, 1, 1, 2),
    ],
)
def test_conv1d_output_shape(
    x_shape, out_channels, kernel_size, stride, padding, dilation
):
    N, C_in, _ = x_shape
    module = nn.Conv1d(
        C_in,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    x = torch.zeros(*x_shape)
    expected = ref_shape(module, x)
    assert (
        conv1d_output_shape(
            x_shape, out_channels, kernel_size, stride, padding, dilation
        )
        == expected
    )


# ---------------------------------------------------------------------------
# conv2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape, out_channels, kernel_size, stride, padding, dilation",
    [
        ((2, 3, 8, 8), 16, 3, 1, 0, 1),
        ((2, 3, 8, 8), 16, 3, 1, 1, 1),
        ((1, 1, 16, 16), 4, 5, 2, 2, 1),
        ((2, 3, 8, 8), 8, 3, 1, "same", 1),
        ((2, 4, 16, 16), 8, 3, 1, 1, 2),
    ],
)
def test_conv2d_output_shape(
    x_shape, out_channels, kernel_size, stride, padding, dilation
):
    N, C_in, _, _ = x_shape
    module = nn.Conv2d(
        C_in,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    x = torch.zeros(*x_shape)
    expected = ref_shape(module, x)
    assert (
        conv2d_output_shape(
            x_shape, out_channels, kernel_size, stride, padding, dilation
        )
        == expected
    )


# ---------------------------------------------------------------------------
# maxpool1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape, kernel_size, stride, padding, dilation, ceil_mode",
    [
        ((2, 4, 16), 3, None, 0, 1, False),
        ((2, 4, 16), 3, 2, 1, 1, False),
        ((2, 4, 16), 3, 2, 0, 1, True),  # ceil_mode
        ((2, 4, 15), 3, 2, 0, 1, False),  # odd length
        ((1, 1, 32), 4, 2, 1, 1, False),
    ],
)
def test_maxpool1d_output_shape(
    x_shape, kernel_size, stride, padding, dilation, ceil_mode
):
    module = nn.MaxPool1d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    x = torch.zeros(*x_shape)
    expected = ref_shape(module, x)
    assert (
        maxpool1d_output_shape(
            x_shape, kernel_size, stride, padding, dilation, ceil_mode
        )
        == expected
    )


# ---------------------------------------------------------------------------
# maxpool2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape, kernel_size, stride, padding, dilation, ceil_mode",
    [
        ((2, 4, 16, 16), 3, None, 0, 1, False),
        ((2, 4, 16, 16), 3, 2, 1, 1, False),
        ((2, 4, 16, 16), (3, 5), 2, 0, 1, False),  # asymmetric kernel
        ((2, 4, 16, 16), 3, 2, 0, 1, True),  # ceil_mode
        ((2, 4, 15, 15), 3, 2, 0, 1, False),  # odd spatial size
    ],
)
def test_maxpool2d_output_shape(
    x_shape, kernel_size, stride, padding, dilation, ceil_mode
):
    module = nn.MaxPool2d(
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    x = torch.zeros(*x_shape)
    expected = ref_shape(module, x)
    assert (
        maxpool2d_output_shape(
            x_shape, kernel_size, stride, padding, dilation, ceil_mode
        )
        == expected
    )


# ---------------------------------------------------------------------------
# adaptiveavgpool2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape, output_size",
    [
        ((2, 4, 16, 16), (4, 4)),
        ((2, 4, 16, 16), (1, 1)),
        ((2, 4, 16, 16), 8),  # int -> square
        ((1, 3, 224, 224), (7, 7)),
    ],
)
def test_adaptiveavgpool2d_output_shape(x_shape, output_size):
    module = nn.AdaptiveAvgPool2d(output_size)
    x = torch.zeros(*x_shape)
    expected = ref_shape(module, x)
    assert adaptiveavgpool2d_output_shape(x_shape, output_size) == expected


# ---------------------------------------------------------------------------
# batchnorm1d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape",
    [
        (4, 16),
        (4, 16, 32),
    ],
)
def test_batchnorm1d_output_shape(x_shape):
    num_features = x_shape[1]
    module = nn.BatchNorm1d(num_features)
    x = torch.randn(*x_shape)
    expected = ref_shape(module, x)
    assert batchnorm1d_output_shape(x_shape, num_features) == expected


# ---------------------------------------------------------------------------
# batchnorm2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape",
    [
        (4, 8, 16, 16),
        (1, 3, 7, 7),
    ],
)
def test_batchnorm2d_output_shape(x_shape):
    num_features = x_shape[1]
    module = nn.BatchNorm2d(num_features)
    x = torch.randn(*x_shape)
    expected = ref_shape(module, x)
    assert batchnorm2d_output_shape(x_shape, num_features) == expected


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape",
    [
        (4, 8),
        (2, 4, 16),
        (2, 4, 8, 8),
    ],
)
def test_relu_output_shape(x_shape):
    x = torch.zeros(*x_shape)
    expected = ref_shape(nn.ReLU(), x)
    assert relu_output_shape(x_shape) == expected


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape",
    [
        (4, 8),
        (2, 4, 8, 8),
    ],
)
def test_sigmoid_output_shape(x_shape):
    x = torch.zeros(*x_shape)
    expected = ref_shape(nn.Sigmoid(), x)
    assert sigmoid_output_shape(x_shape) == expected


# ---------------------------------------------------------------------------
# add  (element-wise, no nn.Module – use tensor addition as reference)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape",
    [
        (4, 8),
        (2, 4, 16),
        (2, 4, 8, 8),
    ],
)
def test_add_output_shape(x_shape):
    x = torch.zeros(*x_shape)
    expected = tuple((x + x).shape)
    assert add_output_shape(x_shape) == expected


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x_shape, start_dim, end_dim",
    [
        ((2, 4, 8, 8), 1, -1),  # default: flatten all but batch
        ((2, 4, 8, 8), 1, 2),  # flatten only first two spatial dims
        ((2, 4, 8, 8), 2, 3),  # flatten only last two dims
        ((2, 4, 8, 8), 0, -1),  # flatten everything
        ((4, 8), 1, -1),  # 2-D input
    ],
)
def test_flatten_output_shape(x_shape, start_dim, end_dim):
    module = nn.Flatten(start_dim=start_dim, end_dim=end_dim)
    x = torch.zeros(*x_shape)
    expected = ref_shape(module, x)
    assert flatten_output_shape(x_shape, start_dim, end_dim) == expected
