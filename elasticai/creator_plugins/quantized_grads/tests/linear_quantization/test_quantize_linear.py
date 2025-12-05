import torch
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.linear_quantization.quantize_linear import quantize_linear_asym_hte, \
    dequantize_linear, quantize_linear_asym_stochastic, quantize_simulated_linear_asym_hte


def test_quantize_linear_hte():
    num_bits = 2
    num_values = Tensor([2 ** num_bits])
    min_value = Tensor([0.])
    max_value = num_values-1
    x = Tensor([-0.1, 0., 10.])

    expected_scale = Tensor([3.3667])
    expected_zero_point = Tensor([-0])
    expected_x = Tensor([0., 0., 3.])

    result_x, result_scale, result_zero_point = quantize_linear_asym_hte(x, min_value, max_value)
    print(f"{result_x}, {result_scale}, {result_zero_point}")
    print(result_scale.values)


    assert torch.equal(result_x, expected_x)
    assert torch.equal(torch.Tensor([result_scale[0]]), expected_scale)
    assert torch.equal(result_zero_point, expected_zero_point)

def test_dequantize_linear():
    x = Tensor([0., 10., 136.])
    scale = Tensor([0.1])
    zero_point = Tensor([3.])

    result_x = dequantize_linear(x, scale, zero_point)
    expected_x = Tensor([3., 4., 16.6])

    assert torch.equal(result_x, expected_x)

def test_fake_quantize_hte():
    min_value = Tensor([0])
    max_value = Tensor([255])
    x = Tensor([-3.2, -2., 10.6])
    print(f"{x=}")
    x = quantize_simulated_linear_asym_hte(x, min_value, max_value)
    print(f"{x=}")
    result_q, scale, zero_point = quantize_linear_asym_hte(x, min_value, max_value)
    result = dequantize_linear(result_q, scale, zero_point)
    print(f"{result=}")
    expected_result = Tensor([-3., -1.9866666793823242, 10.6])
    assert torch.equal(result, expected_result)

def test_quantize_():
    x = Tensor([-10., 10., 136.])

    result_x_1 = quantize_linear_asym_stochastic(x, 256)
    result_x_2 = quantize_linear_asym_stochastic(dequantize_linear(result_x_1[0], result_x_1[1], result_x_1[2]), 256)
    print(result_x_1)
    print(result_x_2)


