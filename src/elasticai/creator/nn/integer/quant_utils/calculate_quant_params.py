import torch

from elasticai.creator.nn.integer._dtypes import FLOAT_DTYPES, INT_DTYPES, check_dtype


def _check_range_dtypes(
    min_float: torch.Tensor,
    max_float: torch.Tensor,
    min_quant: torch.Tensor,
    max_quant: torch.Tensor,
    eps: torch.Tensor,
) -> None:
    check_dtype("min_float", min_float, *FLOAT_DTYPES)
    check_dtype("max_float", max_float, *FLOAT_DTYPES)
    check_dtype("min_quant", min_quant, *INT_DTYPES)
    check_dtype("max_quant", max_quant, *INT_DTYPES)
    check_dtype("eps", eps, *FLOAT_DTYPES)


def calculate_asymmetric_quant_params(
    min_float: torch.Tensor,
    max_float: torch.Tensor,
    min_quant: torch.Tensor,
    max_quant: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate scale factor and zero point for asymmetric quantization.

    The float range `[min_float, max_float]` is mapped onto the integer range
    `[min_quant, max_quant]`.

    Args:
        min_float: Smallest observed float value.
        max_float: Largest observed float value.
        min_quant: Smallest representable integer value.
        max_quant: Largest representable integer value.
        eps: Lower bound for the scale factor, to avoid division by zero.

    Returns:
        A tuple `(scale_factor, zero_point, min_float, max_float)`. The scale
        factor has dtype `torch.float32`, the zero point `torch.int32` and is
        clamped to `[min_quant, max_quant]`. The float range is returned
        unchanged.

    Raises:
        TypeError: If an argument is not a `torch.Tensor`, or if a float
            argument does not have a floating-point dtype or an integer
            argument does not have an integer dtype.
    """
    _check_range_dtypes(min_float, max_float, min_quant, max_quant, eps)

    scale_factor = (max_float - min_float) / (max_quant.float() - min_quant.float())
    scale_factor = torch.max(scale_factor, eps)
    scale_factor = scale_factor.clone().detach().to(torch.float32)

    zero_point = max_quant - (max_float / scale_factor)
    zero_point = zero_point.round_().clamp(min_quant, max_quant)
    zero_point = zero_point.clone().detach().to(torch.int32)

    return scale_factor, zero_point, min_float, max_float


def calculate_symmetric_quant_params(
    min_float: torch.Tensor,
    max_float: torch.Tensor,
    min_quant: torch.Tensor,
    max_quant: torch.Tensor,
    eps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate scale factor and zero point for symmetric quantization.

    The float range is widened to `[-m, m]` with
    `m = max(|min_float|, |max_float|)` and mapped onto the integer range
    `[min_quant, max_quant]`. The zero point is always 0.

    Args:
        min_float: Smallest observed float value.
        max_float: Largest observed float value.
        min_quant: Smallest representable integer value.
        max_quant: Largest representable integer value.
        eps: Lower bound for the scale factor, to avoid division by zero.

    Returns:
        A tuple `(scale_factor, zero_point, min_float, max_float)`. The scale
        factor has dtype `torch.float32`, the zero point is a zero tensor of
        dtype `torch.int32`, and `min_float` and `max_float` are the widened
        symmetric range `-m` and `m`.

    Raises:
        TypeError: If an argument is not a `torch.Tensor`, or if a float
            argument does not have a floating-point dtype or an integer
            argument does not have an integer dtype.
    """
    _check_range_dtypes(min_float, max_float, min_quant, max_quant, eps)

    max_extent = torch.max(torch.abs(min_float), torch.abs(max_float))
    max_float = max_extent
    min_float = -max_extent

    scale_factor = (max_float - min_float) / (max_quant.float() - min_quant.float())
    scale_factor = torch.max(scale_factor, eps).to(torch.float32)

    zero_point = torch.zeros(scale_factor.size(), dtype=torch.int32)

    return scale_factor, zero_point, min_float, max_float
