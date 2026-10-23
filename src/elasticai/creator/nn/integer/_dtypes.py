"""Dtype groups and argument checks shared by the integer modules."""

import torch

FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64)


def check_dtype(name: str, tensor: torch.Tensor, *dtypes: torch.dtype) -> None:
    """Check that an argument is a tensor with one of the allowed dtypes.

    Args:
        name: Argument name used in the error message.
        tensor: The value to check.
        *dtypes: The allowed dtypes, e.g. `torch.int32` or `*FLOAT_DTYPES`.

    Raises:
        TypeError: If `tensor` is not a `torch.Tensor` or its dtype is not
            one of `dtypes`.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype not in dtypes:
        expected = " or ".join(str(dtype) for dtype in dtypes)
        raise TypeError(f"{name} must have dtype {expected}, got {tensor.dtype}")
