import torch
from torch import Tensor

from elasticai.creator_plugins.quantized_grads.linear_quantization.quantize_to_int_with_linear_quantization_style import \
    quantize_to_int_hte_fake, quantize_to_int_hte, quantize_to_int_stochastic


def test_quantize_to_int_hte_fake():
    x = Tensor([1, -5, 3.5, 6, 99999999, -999999999])
    max_val = Tensor([127])
    min_val = Tensor([-128])

    result = quantize_to_int_hte_fake(x, min_val, max_val)

    expected_result = Tensor([1, -5, 4, 6, 127, -128])

    assert torch.equal(result, expected_result)

def test_quantize_to_int_hte():
    device = torch.device("cpu")

    torch.set_default_device(device)

    x = Tensor([1, -5, 3.5, 6, 99999999, -999999999]).to(device)
    max_val = Tensor([127]).to(device)
    min_val = Tensor([-128]).to(device)
    result_x, result_scale, result_offset  = quantize_to_int_hte(x, min_val, max_val)
    expected_result_x = Tensor([1, -5, 4, 6, 127, -128]).to(device)
    expected_result_scale = Tensor([1]).to(device)
    expected_result_offset = Tensor([0]).to(device)
    print(f"{result_scale.device}")
    assert torch.equal(result_x, expected_result_x)
    assert torch.equal(result_scale, expected_result_scale)
    assert torch.equal(result_offset, expected_result_offset)
