import warnings

import torch

from elasticai.creator.nn.integer.quant_utils.simulate_bitshifting import (
    simulate_bitshifting,
)


def test_simulate_bitshifting_basic():
    x_q = torch.tensor([10], dtype=torch.int32)
    m_q_shift = torch.tensor([2], dtype=torch.int32)
    m_q = torch.tensor([4], dtype=torch.int32)
    result = simulate_bitshifting(x_q, m_q_shift, m_q)
    expected_result = torch.tensor([10], dtype=torch.int32)
    assert torch.equal(result, expected_result)


def test_simulate_bitshifting_large_values():
    x_q = torch.tensor([2**30], dtype=torch.int32)
    m_q_shift = torch.tensor([1], dtype=torch.int32)
    m_q = torch.tensor([2], dtype=torch.int32)
    result = simulate_bitshifting(x_q, m_q_shift, m_q)
    expected_result = torch.tensor([2**30], dtype=torch.int32)
    assert torch.equal(result, expected_result)


def test_simulate_bitshifting_small_values():
    x_q = torch.tensor([1], dtype=torch.int32)
    m_q_shift = torch.tensor([1], dtype=torch.int32)
    m_q = torch.tensor([1], dtype=torch.int32)
    result = simulate_bitshifting(x_q, m_q_shift, m_q)
    expected_result = torch.tensor([0], dtype=torch.int32)
    assert torch.equal(result, expected_result)


def test_simulate_bitshifting_negative_values():
    x_q = torch.tensor([-10], dtype=torch.int32)
    m_q_shift = torch.tensor([2], dtype=torch.int32)
    m_q = torch.tensor([4], dtype=torch.int32)
    result = simulate_bitshifting(x_q, m_q_shift, m_q)
    expected_result = torch.tensor([-10], dtype=torch.int32)
    assert torch.eq
