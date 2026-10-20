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


@pytest.mark.parametrize("features", [1, 3, 100, 40000])
@pytest.mark.parametrize("weight_value", [-255, 255])
@pytest.mark.parametrize("bias_value", [None, 12345])
def test_int_mac_preserves_full_accumulation(
    math_ops, features, weight_value, bias_value
):
    inputs = torch.full((2, features), 255, dtype=torch.int32)
    weights = torch.full((1, features), weight_value, dtype=torch.int32)
    bias = None if bias_value is None else torch.tensor([bias_value], dtype=torch.int32)

    result = math_ops.int_mac(inputs, weights, 8, 8, bias)

    expected = features * 255 * weight_value + (bias_value or 0)
    expected_dtype = torch.int64 if features == 40000 else torch.int32
    torch.testing.assert_close(result, torch.full((2, 1), expected, dtype=expected_dtype))


def test_intadd_keeps_wide_result_until_requantization(math_ops):
    inputs = torch.tensor([2_000_000_000], dtype=torch.int32)

    result = math_ops.intadd(inputs, inputs, 33)

    torch.testing.assert_close(
        result, torch.tensor([4_000_000_000], dtype=torch.int64)
    )
    torch.testing.assert_close(
        math_ops.clamp_result(result, 8), torch.tensor([127], dtype=torch.int32)
    )
