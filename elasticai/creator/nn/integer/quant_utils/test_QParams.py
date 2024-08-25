import torch

from elasticai.creator.nn.integer.quant_utils.Observers import GlobalMinMaxObserver
from elasticai.creator.nn.integer.quant_utils.QParams import (
    AsymmetricSignedQParams,
    AsymmetricUnsignedQParams,
    SymmetricSignedQParams,
    SymmetricUnsignedQParams,
)


def test_AsymmetricSignedQParams():
    ASSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([[-2.0, 0.5, 1.0, 2.0, 3.0]], dtype=torch.float32)
    min_quant = torch.tensor([-128], dtype=torch.int32)
    max_quant = torch.tensor([127], dtype=torch.int32)
    scale_factor = torch.tensor([0.0196078431372549], dtype=torch.float32)
    zero_point = torch.tensor([-26], dtype=torch.int32)

    ASSSParams.update_quant_params(x)
    actual_q_x = ASSSParams.quantize(x)
    expected_q_x = torch.tensor([[-128, 0, 25, 76, 127]], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = ASSSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [[-2.0000, 0.5098, 1.0000, 2.0000, 3.0000]], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-10)


def test_AsymmetricUnsignedQParams():
    ASUSParams = AsymmetricUnsignedQParams(
        quant_bits=8, observer=GlobalMinMaxObserver()
    )

    x = torch.tensor([-2.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32)
    min_quant = torch.tensor([0], dtype=torch.int32)
    max_quant = torch.tensor([255], dtype=torch.int32)
    scale_factor = torch.tensor([0.0196078431372549], dtype=torch.float32)
    zero_point = torch.tensor([102], dtype=torch.int32)

    ASUSParams.update_quant_params(x)
    actual_q_x = ASUSParams.quantize(x)
    expected_q_x = torch.tensor([0, 128, 153, 204, 255], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = ASUSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [-2.0000, 0.5098, 1.0000, 2.0000, 3.0000], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


def test_SymmetricSignedQParams():
    SSSSParams = SymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())
    x = torch.tensor([-2.0, 0.5, 0.5, 1.0, 2.0], dtype=torch.float32)
    min_quant = torch.tensor([-127], dtype=torch.int32)
    max_quant = torch.tensor([127], dtype=torch.int32)
    scale_factor = torch.tensor([0.0157480314], dtype=torch.float32)
    zero_point = torch.tensor([0], dtype=torch.int32)

    SSSSParams.update_quant_params(x)
    actual_q_x = SSSSParams.quantize(x)
    expected_q_x = torch.tensor([-127, 32, 32, 64, 127], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = SSSSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [-2.0000, 0.5039, 0.5039, 1.0079, 2.0000], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


def test_SymmetricUnsignedQParams():
    # BUG:  this scheme lead to huge quantization error
    SSUSParams = SymmetricUnsignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 0.5, 1.0, 2.0], dtype=torch.float32)
    min_quant = torch.tensor([0], dtype=torch.int32)
    max_quant = torch.tensor([254], dtype=torch.int32)
    scale_factor = torch.tensor([0.0157480314], dtype=torch.float32)
    zero_point = torch.tensor([0], dtype=torch.int32)

    SSUSParams.update_quant_params(x)
    actual_q_x = SSUSParams.quantize(x)
    expected_q_x = torch.tensor([0, 32, 32, 64, 127], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)

    actual_dq_x = SSUSParams.dequantize(actual_q_x)
    expected_dq_x = torch.tensor(
        [0.0000, 0.5039, 0.5039, 1.0079, 2.0000], dtype=torch.float32
    )
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


def test_set_scale_factor():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    scale_factor = torch.tensor([0.0157480314], dtype=torch.float32)
    SSSSParams.set_scale_factor(scale_factor)

    assert torch.equal(SSSSParams.scale_factor, scale_factor)


def test_set_zero_point():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    zero_point = torch.tensor([12], dtype=torch.int32)
    SSSSParams.set_zero_point(zero_point)

    assert torch.equal(SSSSParams.zero_point, zero_point)


def test_set_quant_range():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    SSSSParams.set_quant_range(16)

    assert SSSSParams.quant_bits == 16
    assert torch.equal(SSSSParams.min_quant, torch.tensor([-32768], dtype=torch.int32))
    assert torch.equal(SSSSParams.max_quant, torch.tensor([32767], dtype=torch.int32))


def test_quantize():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32)
    SSSSParams.update_quant_params(x)

    actual_q_x = SSSSParams.quantize(x)
    expected_q_x = torch.tensor([-128, 84, 127], dtype=torch.int32)
    assert torch.equal(actual_q_x, expected_q_x)


def test_dequantize():
    SSSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32)
    SSSSParams.update_quant_params(x)
    q_x = SSSSParams.quantize(x)
    actual_dq_x = SSSSParams.dequantize(q_x)
    expected_dq_x = torch.tensor([-2.0000, 0.4941, 1.0000], dtype=torch.float32)
    assert torch.allclose(actual_dq_x, expected_dq_x, atol=1e-4)


def test_state_dict():
    ASSSParams = AsymmetricSignedQParams(quant_bits=8, observer=GlobalMinMaxObserver())

    x = torch.tensor([-2.0, 0.5, 1.0], dtype=torch.float32)
    ASSSParams.update_quant_params(x)

    state_dict = ASSSParams.state_dict()

    ASSSParams_new = AsymmetricSignedQParams(
        quant_bits=8, observer=GlobalMinMaxObserver()
    )
    ASSSParams_new.load_state_dict(state_dict)

    assert torch.equal(ASSSParams.min_float, ASSSParams_new.min_float)
    assert torch.equal(ASSSParams.max_float, ASSSParams_new.max_float)
    assert torch.equal(ASSSParams.scale_factor, ASSSParams_new.scale_factor)
    assert torch.equal(ASSSParams.zero_point, ASSSParams_new.zero_point)
    assert torch.equal(ASSSParams.min_quant, ASSSParams_new.min_quant)
    assert torch.equal(ASSSParams.max_quant, ASSSParams_new.max_quant)
