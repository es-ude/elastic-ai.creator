import warnings

import torch

from elasticai.creator.nn.integer._dtypes import INT_DTYPES, check_dtype


def simulate_bitshifting(
    x_q: torch.Tensor, m_q_shift: torch.Tensor, m_q: torch.Tensor
) -> torch.Tensor:
    """Approximate a multiplication by a float scale factor with integers.

    Computes `round(x_q * m_q / 2**m_q_shift)`, which is how the hardware
    applies a scale factor `M ≈ m_q / 2**m_q_shift` (see `scaling_M`).
    The product is computed in `torch.int64`.

    Args:
        x_q: Integer tensor to scale.
        m_q_shift: Number of bits to shift right; must hold a single value.
        m_q: Integer multiplier.

    Returns:
        The scaled tensor with dtype `torch.int32`.

    Ties are rounded to the nearest even value, like `torch.round`. A
    warning is issued if `x_q * m_q` would overflow `torch.int64`.

    Raises:
        TypeError: If an argument is not a `torch.Tensor` or does not have an
            integer dtype.
    """
    check_dtype("x_q", x_q, *INT_DTYPES)
    check_dtype("m_q_shift", m_q_shift, *INT_DTYPES)
    check_dtype("m_q", m_q, *INT_DTYPES)

    # TODO: check if torch.int64 could be downgraded to torch.int32
    x_q = x_q.to(torch.int64)
    m_q = m_q.to(torch.int64).to(x_q.device)
    shift = int(m_q_shift.item())

    # int64 multiplication wraps around silently, so check the bound before
    # multiplying instead of inspecting the product afterwards.
    max_factor = torch.iinfo(torch.int64).max // m_q.abs().clamp(min=1)
    if torch.any(x_q.abs() > max_factor):
        warnings.warn("Overflow of product in simulate_bitshifting", stacklevel=2)

    product = x_q * m_q
    if shift <= 0:
        return product.int()

    # BUG: The generated hardware rounds differently. The VHDL templates (e.g.
    #   `linear.tpl.vhd`) shift the magnitude `|product|` right and add 1 if
    #   bit `shift - 1` is set, i.e. they round half away from zero. Results
    #   therefore differ from the hardware on exact ties, e.g. 2.5 -> 2 here
    #   but 3 in hardware, and -2.5 -> -2 here but -3 in hardware.
    #   Possible solution: replace the rounding below with
    #   `torch.sign(product) * ((product.abs() + half) >> shift)` and update
    #   the tests that expect round-half-to-even.
    # NOTE: Integer right shift with round-half-to-even, matching torch.round, but
    #   without the precision loss of dividing in floating point.
    quotient = product >> shift
    remainder = product - (quotient << shift)
    half = 1 << (shift - 1)
    round_up = (remainder > half) | ((remainder == half) & (quotient % 2 == 1))
    return (quotient + round_up.to(torch.int64)).int()
