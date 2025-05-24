import torch


def scaling_M(M: torch.FloatTensor, m_q_shift_limit=32):
    assert torch.all(M > 0), "M should be positive"
    M = torch.round(M * 10**5) / 10**5
    m_q_shift = torch.tensor(1, dtype=torch.int32)

    # "if M is approaching 0 or equal to 0, we will give up approximating it"
    m_q = torch.tensor(0, dtype=torch.int32)

    if torch.any(M != 0):
        while m_q_shift.item() < m_q_shift_limit:
            m_q = torch.round(M * (2**m_q_shift)).type(torch.int32)
            error = (M - m_q * (2 ** (-m_q_shift.item()))) / M

            if torch.any(torch.isnan(error)):
                raise ValueError(
                    "Error contains NaN values, check input M or computation logic."
                )

            if torch.all(error > 0.0001) or torch.all(error < 0):
                m_q_shift += 1
            else:
                break

    return m_q_shift, m_q
