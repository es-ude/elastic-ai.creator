import warnings

import torch


def simulate_bitshifting(
    x_q: torch.IntTensor, M_q_shift: torch.IntTensor, M_q: torch.IntTensor
):
    # TODO: check if torch.int64 could be downgraded to torch.int32
    x_q = x_q.to(torch.int64)
    M_q_shift = M_q_shift.to(x_q.device)
    M_q = M_q.to(torch.int64).to(x_q.device)

    product = x_q * M_q
    max_int64 = torch.iinfo(torch.int64).max
    min_int64 = torch.iinfo(torch.int64).min
    if torch.any(product > max_int64) or torch.any(product < min_int64):
        warnings.warn("Overflow of product in simulate_bitshifting")

    approx_result = (product / (2 ** M_q_shift.item())).round_().int()

    return approx_result
