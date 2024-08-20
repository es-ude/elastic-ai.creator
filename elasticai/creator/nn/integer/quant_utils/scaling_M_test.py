import torch

from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M


def test_scaling_M_basic():
    M = torch.tensor([0.12345], dtype=torch.float32)
    m_q_shift, m_q = scaling_M(M)

    assert m_q_shift.item() > 0
    assert m_q_shift.item() <= 32

    expected_m_q = torch.round(M * (2 ** m_q_shift.item())).type(torch.int32)
    torch.testing.assert_close(m_q, expected_m_q, rtol=1e-5, atol=1e-8)

    recovered_M = m_q * (2 ** (-m_q_shift.item()))
    error = (M - recovered_M) / M
    assert torch.all(error < 0.0001)
