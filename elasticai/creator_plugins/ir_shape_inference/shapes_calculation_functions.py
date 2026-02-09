import math
from typing import Literal, Tuple, Union

Padding2D = Union[int, Tuple[int, int], Literal["same", "valid"]]
IntOr2 = Union[int, Tuple[int, int]]
Padding = Union[int, Literal["same", "valid"]]


def linear_output_shape(N: int, out_features: int) -> tuple[int, int]:
    return N, out_features


def conv1d_out_len(
    L_in: int,
    kernel_size: int,
    stride: int = 1,
    padding: Padding = 0,
    dilation: int = 1,
) -> Tuple[int, Tuple[int, int]]:
    """
    Returns (L_out, (pad_left, pad_right)) consistent with PyTorch Conv1d.

    - padding="valid": pad_left=pad_right=0
    - padding="same": requires stride==1 in PyTorch; returns asymmetric padding if needed
    - padding=int: symmetric padding on both sides
    """
    if L_in < 0:
        raise ValueError("L_in must be non-negative")
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if dilation <= 0:
        raise ValueError("dilation must be positive")

    # "valid" == no padding
    if padding == "valid":
        p = 0
        L_out = math.floor(
            (L_in + 2 * p - dilation * (kernel_size - 1) - 1) / stride + 1
        )
        return L_out, (0, 0)

    # "same" keeps output length == input length (PyTorch Conv1d requires stride==1)
    if padding == "same":
        if stride != 1:
            raise ValueError(
                'PyTorch Conv1d padding="same" does not support stride != 1.'
            )
        L_out = L_in
        total_pad = dilation * (kernel_size - 1)  # amount needed to keep length
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left  # may be asymmetric when total_pad is odd
        return L_out, (pad_left, pad_right)

    # numeric padding (symmetric)
    p = int(padding)
    if p < 0:
        raise ValueError("numeric padding must be >= 0")

    L_out = math.floor((L_in + 2 * p - dilation * (kernel_size - 1) - 1) / stride + 1)
    return L_out, (p, p)


def conv1d_output_shape(
    x_shape: Tuple[int, int, int],  # (N, C_in, L_in)
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: Padding = 0,
    dilation: int = 1,
) -> Tuple[int, int, int]:
    """
    Returns (N, C_out, L_out) for Conv1d.
    """
    N, C_in, L_in = x_shape
    L_out, _ = conv1d_out_len(
        L_in=L_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return (N, out_channels, L_out)


def _to_2tuple(v: IntOr2) -> Tuple[int, int]:
    return (v, v) if isinstance(v, int) else v


def conv2d_out_hw(
    H_in: int,
    W_in: int,
    kernel_size: IntOr2,
    stride: IntOr2 = 1,
    padding: Padding2D = 0,
    dilation: IntOr2 = 1,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Returns:
      (H_out, W_out),
      pad_tuple_for_F_pad = (pad_left, pad_right, pad_top, pad_bottom)

    Matches PyTorch Conv2d/F.conv2d shape rules.

    Notes:
    - padding="valid" -> no padding
    - padding="same"  -> requires stride==(1,1) in PyTorch; may be asymmetric
    - numeric padding -> symmetric per-dimension: (padH, padW)
    """
    if H_in < 0 or W_in < 0:
        raise ValueError("H_in and W_in must be non-negative")

    kH, kW = _to_2tuple(kernel_size)
    sH, sW = _to_2tuple(stride)
    dH, dW = _to_2tuple(dilation)

    if kH <= 0 or kW <= 0:
        raise ValueError("kernel_size must be positive")
    if sH <= 0 or sW <= 0:
        raise ValueError("stride must be positive")
    if dH <= 0 or dW <= 0:
        raise ValueError("dilation must be positive")

    # "valid" == no padding
    if padding == "valid":
        pH = pW = 0
        H_out = math.floor((H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1)
        W_out = math.floor((W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1)
        return (H_out, W_out), (0, 0, 0, 0)

    # "same" keeps output size == input size (PyTorch requires stride == 1)
    if padding == "same":
        if (sH, sW) != (1, 1):
            raise ValueError(
                'PyTorch Conv2d padding="same" does not support stride != 1.'
            )
        H_out, W_out = H_in, W_in

        total_pad_h = dH * (kH - 1)
        total_pad_w = dW * (kW - 1)

        pad_top = total_pad_h // 2
        pad_bottom = total_pad_h - pad_top
        pad_left = total_pad_w // 2
        pad_right = total_pad_w - pad_left

        return (H_out, W_out), (pad_left, pad_right, pad_top, pad_bottom)

    # numeric padding (symmetric per dimension)
    if isinstance(padding, int):
        pH = pW = padding
    else:
        pH, pW = padding

    if pH < 0 or pW < 0:
        raise ValueError("numeric padding must be >= 0")

    H_out = math.floor((H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1)
    W_out = math.floor((W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1)

    return (H_out, W_out), (pW, pW, pH, pH)


def conv2d_output_shape(
    x_shape: Tuple[int, int, int, int],  # (N, C_in, H_in, W_in)
    out_channels: int,
    kernel_size: IntOr2,
    stride: IntOr2 = 1,
    padding: Padding2D = 0,
    dilation: IntOr2 = 1,
) -> Tuple[int, int, int, int]:
    """
    Returns (N, C_out, H_out, W_out) for Conv2d.
    """
    N, C_in, H_in, W_in = x_shape
    (H_out, W_out), _ = conv2d_out_hw(
        H_in=H_in,
        W_in=W_in,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    if H_out < 0 or W_out < 0:
        raise ValueError(
            f"Computed negative output size: (H_out,W_out)=({H_out},{W_out})."
        )
    return (N, out_channels, H_out, W_out)


def maxpool1d_output_shape(
    x_shape: Tuple[int, int, int],  # (N, C, L_in)
    kernel_size: int,
    stride: int | None = None,  # defaults to kernel_size
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
) -> Tuple[int, int, int]:
    """Returns (N, C, L_out) for MaxPool1d."""
    N, C, L_in = x_shape
    if stride is None:
        stride = kernel_size
    if ceil_mode:
        L_out = math.ceil(
            (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
    else:
        L_out = math.floor(
            (L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        )
    return (N, C, L_out)


def maxpool2d_output_shape(
    x_shape: Tuple[int, int, int, int],  # (N, C, H_in, W_in)
    kernel_size: IntOr2,
    stride: IntOr2 | None = None,  # defaults to kernel_size
    padding: IntOr2 = 0,
    dilation: IntOr2 = 1,
    ceil_mode: bool = False,
) -> Tuple[int, int, int, int]:
    """Returns (N, C, H_out, W_out) for MaxPool2d."""
    N, C, H_in, W_in = x_shape
    kH, kW = _to_2tuple(kernel_size)
    sH, sW = (kH, kW) if stride is None else _to_2tuple(stride)
    pH, pW = _to_2tuple(padding)
    dH, dW = _to_2tuple(dilation)
    if ceil_mode:
        H_out = math.ceil((H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1)
        W_out = math.ceil((W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1)
    else:
        H_out = math.floor((H_in + 2 * pH - dH * (kH - 1) - 1) / sH + 1)
        W_out = math.floor((W_in + 2 * pW - dW * (kW - 1) - 1) / sW + 1)
    return (N, C, H_out, W_out)


def adaptiveavgpool2d_output_shape(
    x_shape: Tuple[int, int, int, int],  # (N, C, H_in, W_in)
    output_size: Union[int, Tuple[int, int]],
) -> Tuple[int, int, int, int]:
    """Returns (N, C, H_out, W_out) for AdaptiveAvgPool2d."""
    N, C, _, _ = x_shape
    H_out, W_out = _to_2tuple(output_size)
    return (N, C, H_out, W_out)


def batchnorm1d_output_shape(
    x_shape: Tuple,
    num_features: int,
) -> Tuple:
    """Returns the same shape as input (BatchNorm1d is shape-preserving)."""
    return x_shape


def batchnorm2d_output_shape(
    x_shape: Tuple[int, int, int, int],
    num_features: int,
) -> Tuple[int, int, int, int]:
    """Returns the same shape as input (BatchNorm2d is shape-preserving)."""
    return x_shape


def relu_output_shape(x_shape: Tuple) -> Tuple:
    """Returns the same shape as input (ReLU is shape-preserving)."""
    return x_shape


def sigmoid_output_shape(x_shape: Tuple) -> Tuple:
    """Returns the same shape as input (Sigmoid is shape-preserving)."""
    return x_shape


def add_output_shape(x_shape: Tuple) -> Tuple:
    """Returns the same shape as input (element-wise Add is shape-preserving)."""
    return x_shape


def flatten_output_shape(
    x_shape: Tuple,
    start_dim: int = 1,
    end_dim: int = -1,
) -> Tuple:
    """Returns the output shape of nn.Flatten."""
    ndim = len(x_shape)
    s = start_dim if start_dim >= 0 else ndim + start_dim
    e = end_dim if end_dim >= 0 else ndim + end_dim
    flat_size = 1
    for i in range(s, e + 1):
        flat_size *= x_shape[i]
    return tuple(x_shape[:s]) + (flat_size,) + tuple(x_shape[e + 1 :])
