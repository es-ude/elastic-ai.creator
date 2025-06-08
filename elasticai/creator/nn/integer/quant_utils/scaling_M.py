import torch


def scaling_M(
    M: torch.FloatTensor,
    error_threshold: float = None,
    m_q_shift_threshold: int = None,
):
    assert torch.all(M > 0), "M should be positive"
    M = torch.round(M * 10**5) / 10**5
    m_q_shift = torch.tensor(1, dtype=torch.int32)
    m_q = torch.tensor(
        0, dtype=torch.int32
    )  # "if M is approaching 0 or equal to 0, we will give up approximating it"

    m_q_shift_threshold = 32 if m_q_shift_threshold is None else m_q_shift_threshold
    error_threshold = 0.0001 if error_threshold is None else error_threshold
    if torch.any(M != 0):
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
