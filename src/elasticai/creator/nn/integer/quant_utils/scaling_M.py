import torch

from elasticai.creator.nn.integer._dtypes import FLOAT_DTYPES, check_dtype


def scaling_M(
    M: torch.Tensor,
    error_threshold: float | None = None,
    m_q_shift_threshold: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate a positive float scale factor as `m_q / 2**m_q_shift`.

    `M` is rounded to 5 decimal places. Starting at `m_q_shift = 1`, the shift
    is increased until the relative error of the approximation is at most
    `error_threshold` or the shift reaches `m_q_shift_threshold`.

    Args:
        M: The positive scale factor to approximate.
        error_threshold: Maximum allowed relative error. Defaults to 0.0001.
        m_q_shift_threshold: Upper limit for the shift. Defaults to 32.

    Returns:
        A tuple `(m_q_shift, m_q)` of `torch.int32` tensors.

    Raises:
        TypeError: If `M` is not a `torch.Tensor` or does not have a
            floating-point dtype.
        ValueError: If `M` is not positive, if it is smaller than 0.000005 and
            therefore rounds to 0, or if the relative error becomes NaN.
    """
    check_dtype("M", M, *FLOAT_DTYPES)
    if not torch.all(M > 0):
        raise ValueError(f"M must be positive, got {M}")
    M = torch.round(M * 10**5) / 10**5
    # A scale factor of 0 would silently set every output of the layer to 0.
    if torch.any(M == 0):
        raise ValueError(
            f"M must be at least 0.000005 to be approximated, got {M}. Check the "
            "scale factors of the layer's inputs, weights and outputs."
        )
    m_q_shift = torch.tensor(1, dtype=torch.int32)
    m_q = torch.tensor(0, dtype=torch.int32)

    m_q_shift_threshold = 32 if m_q_shift_threshold is None else m_q_shift_threshold
    error_threshold = 0.0001 if error_threshold is None else error_threshold
    while m_q_shift.item() < m_q_shift_threshold:
        m_q = torch.round(M * (2**m_q_shift)).type(torch.int32)
        error = (M - m_q * (2 ** (-m_q_shift.item()))) / M

        if torch.any(torch.isnan(error)):
            raise ValueError(
                "Error contains NaN values, check input M or computation logic."
            )
        if torch.all(torch.abs(error) > error_threshold):
            m_q_shift += 1
        else:
            break
    return m_q_shift, m_q
