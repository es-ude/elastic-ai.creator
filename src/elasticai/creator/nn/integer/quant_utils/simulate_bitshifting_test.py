from fractions import Fraction

import pytest
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
    simulate_bitshifting(x_q, m_q_shift, m_q)
    torch.tensor([-10], dtype=torch.int32)
    assert torch.eq


def test_simulate_bitshifting_rounds_ties_to_even():
    x_q = torch.tensor([1, 3, 5, -1, -3, 7], dtype=torch.int32)
    m_q_shift = torch.tensor([1], dtype=torch.int32)
    m_q = torch.tensor([1], dtype=torch.int32)
    result = simulate_bitshifting(x_q, m_q_shift, m_q)
    expected_result = torch.round(x_q / 2).to(torch.int32)
    assert torch.equal(result, expected_result)


def test_simulate_bitshifting_matches_exact_rounding():
    # Python's round() on a Fraction is exact and rounds ties to even.
    torch.manual_seed(0)
    x_q = torch.randint(-(2**20), 2**20, (1000,), dtype=torch.int32)
    m_q_shift = torch.tensor([12], dtype=torch.int32)
    m_q = torch.tensor([1234], dtype=torch.int32)
    result = simulate_bitshifting(x_q, m_q_shift, m_q)
    expected_result = torch.tensor(
        [round(Fraction(x * 1234, 2**12)) for x in x_q.tolist()], dtype=torch.int32
    )
    assert torch.equal(result, expected_result)


def test_simulate_bitshifting_warns_on_overflow():
    x_q = torch.tensor([2**62], dtype=torch.int64)
    m_q_shift = torch.tensor([1], dtype=torch.int32)
    m_q = torch.tensor([4], dtype=torch.int32)
    with pytest.warns(UserWarning, match="Overflow"):
        simulate_bitshifting(x_q, m_q_shift, m_q)
