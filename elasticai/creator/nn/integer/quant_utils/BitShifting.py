import warnings

import torch


def scaling_M(M: torch.FloatTensor):
    M = torch.round(M * 10**5) / 10**5
    M_q_shift = torch.tensor(1, dtype=torch.int32)

    while M_q_shift.item() < 32:
        # Attention: take care of the bitwidth of the scaler in the vdhl template
        M_q = torch.round(M * (2**M_q_shift)).type(torch.int32)

        error = (M - M_q * (2 ** (-M_q_shift.item()))) / M
        if torch.all(error > 0.0001) or torch.all(error < 0):
            M_q_shift += 1
        else:
            break
    return M_q_shift, M_q


def simulate_bitshifting(x_q: torch.Tensor, M_q_shift: torch.Tensor, M_q: torch.Tensor):
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
