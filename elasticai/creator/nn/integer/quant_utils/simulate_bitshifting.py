import warnings

import torch


def simulate_bitshifting(
    x_q: torch.IntTensor, m_q_shift: torch.IntTensor, m_q: torch.IntTensor
):
    # TODO: check if torch.int64 could be downgraded to torch.int32
    x_q = x_q.to(torch.int64)
    m_q_shift = m_q_shift.to(x_q.device)
    m_q = m_q.to(torch.int64).to(x_q.device)

    product = x_q * m_q
    max_int64 = torch.iinfo(torch.int64).max
    min_int64 = torch.iinfo(torch.int64).min
    if torch.any(product > max_int64) or torch.any(product < min_int64):
        warnings.warn("Overflow of product in simulate_bitshifting")

    approx_result = (product / (2 ** m_q_shift.item())).round_().int()

    return approx_result
