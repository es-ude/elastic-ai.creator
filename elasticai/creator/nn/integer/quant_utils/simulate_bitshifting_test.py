import warnings

import torch


def scaling_M(m: torch.Tensor):
    N_shifts = torch.tensor(1, dtype=torch.int32)
    m = torch.round(m * 10**5) / 10**5

    while N_shifts.item() < 32:
        # BUG: there is no limitation of m_int's range
        m_int = torch.round(m * (2**N_shifts)).type(torch.int32)
        error = (m - m_int * (2 ** (-N_shifts.item()))) / m
        if torch.all(error > 0.0001) or torch.all(error < 0):
            N_shifts += 1
        else:
            break
    return N_shifts, m_int


def simulate_bitshifting(
    x_q: torch.Tensor, N_shifts: torch.Tensor, m_int: torch.Tensor
):
    x_q = x_q.to(torch.int64)
    N_shifts = N_shifts.to(x_q.device)
    m_int = m_int.to(torch.int64).to(x_q.device)

    product = x_q * m_int
    max_int64 = torch.iinfo(torch.int64).max
    min_int64 = torch.iinfo(torch.int64).min
    if torch.any(product > max_int64) or torch.any(product < min_int64):
        warnings.warn("Overflow of product in simulate_bitshifting")

    approx_result = (product / (2 ** N_shifts.item())).round_().int()

    return approx_result
