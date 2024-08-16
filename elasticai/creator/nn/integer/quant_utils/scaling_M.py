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
