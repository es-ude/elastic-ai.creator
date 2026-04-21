import pytest
import torch

from elasticai.creator.nn.integer.math_operations.math_operations import MathOperations


@pytest.fixture
def math_ops():
    return MathOperations()


def test_clamp_result_do_clamp(math_ops):
    quant_bits = 8
    inputs = torch.tensor([128, 34, 56], dtype=torch.int32)
    actual_results = math_ops.clamp_result(inputs, quant_bits)
    expected_results = torch.tensor([127, 34, 56], dtype=torch.int32)
    assert torch.equal(actual_results, expected_results)


def test_clamp_result_do_not_clamp(math_ops):
    quant_bits = 8
    inputs = torch.tensor([127, 34, 56], dtype=torch.int32)
    actual_results = math_ops.clamp_result(inputs, quant_bits)
    expected_results = torch.tensor([127, 34, 56], dtype=torch.int32)
    assert torch.equal(actual_results, expected_results)


def test_intadd_correctly(math_ops):
    inputs_a = torch.tensor([127, 34, 89], dtype=torch.int32)
    inputs_b = torch.tensor([-54, 29, 39], dtype=torch.int32)
    c_quant_bits = 9
    actual_results = math_ops.intadd(inputs_a, inputs_b, c_quant_bits)
    expected_results = torch.tensor([73, 63, 128], dtype=torch.int32)
    assert torch.equal(actual_results, expected_results)


def test_intadd_with_wrong_dtype(math_ops):
    inputs_a = torch.tensor([127, 34, 89], dtype=torch.float32)
    inputs_b = torch.tensor([-54, 29, 39], dtype=torch.int32)
    c_quant_bits = 9
    with pytest.raises(AssertionError):
        math_ops.intadd(inputs_a, inputs_b, c_quant_bits)


def test_intsub_correctly(math_ops):
    inputs_a = torch.tensor([127, 34, 89], dtype=torch.int32)
    inputs_b = torch.tensor([-54, 29, 39], dtype=torch.int32)
    c_quant_bits = 9
    actual_results = math_ops.intsub(inputs_a, inputs_b, c_quant_bits)
    expected_results = torch.tensor([181, 5, 50], dtype=torch.int32)
    assert torch.equal(actual_results, expected_results)


def test_intsub_with_wrong_dtype(math_ops):
    inputs_a = torch.tensor([127, 34, 89], dtype=torch.int32)
    inputs_b = torch.tensor([-54, 29, 39], dtype=torch.float32)
    c_quant_bits = 9
    with pytest.raises(AssertionError):
        math_ops.intsub(inputs_a, inputs_b, c_quant_bits)


def test_intmatmul_correctly():
    inputs_a = torch.tensor([[127, 34], [56, 78]], dtype=torch.int32)
    inputs_b = torch.tensor([[-54, 29], [39, 12]], dtype=torch.int32)
    c_quant_bits = 18

    math_ops = MathOperations()
    actual_results = math_ops.intmatmul(inputs_a, inputs_b, c_quant_bits)
    expected_results = torch.tensor(
        [
            [127 * -54 + 34 * 39, 127 * 29 + 34 * 12],
            [56 * -54 + 78 * 39, 56 * 29 + 78 * 12],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(actual_results, expected_results)


def test_intmatmul_with_wrong_dtype():
    inputs_a = torch.tensor([[127, 34], [56, 78]], dtype=torch.int32)
    inputs_b = torch.tensor([[-54, 29], [39, 12]], dtype=torch.float32)
    c_quant_bits = 18

    math_ops = MathOperations()
    with pytest.raises(AssertionError):
        math_ops.intmatmul(inputs_a, inputs_b, c_quant_bits)
