import torch

from elasticai.creator.nn.integer.quant_utils.scaling_M import scaling_M


def test_scaling_M_basic():
    M = torch.tensor([0.12345], dtype=torch.float32)
    m_q_shift, m_q = scaling_M(M)
    approx_M = m_q * (2 ** (-m_q_shift.item()))
    error = torch.abs((M - approx_M) / M)
    assert torch.all(error < 0.0001)


def test_scaling_M_large_value():
    M = torch.tensor([1000.0], dtype=torch.float32)
    m_q_shift, m_q = scaling_M(M)
    approx_M = m_q * (2 ** (-m_q_shift.item()))
    error = torch.abs((M - approx_M) / M)
    assert torch.all(error < 0.0001)


def test_scaling_M_small_value():
    M = torch.tensor([0.0001], dtype=torch.float32)
    m_q_shift, m_q = scaling_M(M)
    approx_M = m_q * (2 ** (-m_q_shift.item()))
    error = torch.abs((M - approx_M) / M)
    assert torch.all(error < 0.0001)


def test_scaling_M_shift_limit():
    M = torch.tensor([0.5], dtype=torch.float32)
    m_q_shift, m_q = scaling_M(M, m_q_shift_limit=5)
    approx_M = m_q * (2 ** (-m_q_shift.item()))
    error = torch.abs((M - approx_M) / M)
    assert torch.all(error < 0.0001)
