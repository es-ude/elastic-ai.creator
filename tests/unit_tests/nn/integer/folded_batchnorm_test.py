import pytest
import torch
from torch import nn

from src.elasticai.creator.nn.integer.conv1dbn import Conv1dBN
from torch.nn import functional as F

def test_conv1dbn_matches_pytorch_bn():
    torch.manual_seed(0)

    B = 8
    C_in = 4
    C_out = 7
    L = 32

    expected_conv = nn.Conv1d(C_in, C_out, kernel_size=3, padding=1)
    expected_bn = nn.BatchNorm1d(C_out)

    qat = Conv1dBN(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=3,
        seq_len=L,
        padding=1,
        quant_bits=8,
        bias_bits=32,
        activation_bits=16,
        device="cpu",
    )

    qat.conv1d.load_state_dict(expected_conv.state_dict())
    qat.bn1d.load_state_dict(expected_bn.state_dict())

    qat.train()
    expected_conv.train()
    expected_bn.train()

    for _ in range(20):
        x = torch.randn(B, C_in, L)
        y_expected = expected_bn(expected_conv(x))
        y_qat = qat(x, enable_simquant=False)

        torch.testing.assert_close(
            y_qat,
            y_expected,
            rtol=1e-5,
            atol=1e-5,
        )

    torch.testing.assert_close(
        qat.bn1d.running_mean,
        expected_bn.running_mean,
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        qat.bn1d.running_var,
        expected_bn.running_var,
        rtol=1e-5,
        atol=1e-5,
    )


def setup_conv1dBN(C_IN, C_OUT, K, L):
    layer = Conv1dBN(
        in_channels=C_IN,
        out_channels=C_OUT,
        kernel_size=K,
        seq_len=L,
        padding=1,
        quant_bits=8,
        bias_bits=32,
        activation_bits=16,
        device="cpu",
    )

    layer.eval()

    # Give Conv and BN deliberately non-trivial parameters.
    with torch.no_grad():
        layer.conv1d.weight.copy_(torch.randn_like(layer.conv1d.weight))
        layer.conv1d.bias.copy_(torch.randn_like(layer.conv1d.bias))
        layer.bn1d.weight.copy_(torch.randn_like(layer.bn1d.weight))
        layer.bn1d.bias.copy_(torch.randn_like(layer.bn1d.bias))
        layer.bn1d.running_mean.copy_(torch.randn_like(layer.bn1d.running_mean))
        layer.bn1d.running_var.copy_(torch.rand_like(layer.bn1d.running_var) + 0.5)
    return layer
def calc_expected_without_folding(x, conv1d_bn):
    with torch.no_grad():
        expected = conv1d_bn.bn1d(conv1d_bn.conv1d(x))
    return expected
def calc_actual_with_folding(x, conv1d_bn):
    fused_weight, fused_bias = conv1d_bn._fold_bn()

    # Run the equivalent folded convolution.
    with torch.no_grad():
        actual =F.conv1d(
            x,
            fused_weight,
            fused_bias,
            stride=conv1d_bn.conv1d.stride,
            padding=conv1d_bn.conv1d.padding,
            dilation=conv1d_bn.conv1d.dilation,
            groups=conv1d_bn.conv1d.groups,
        )
    return actual

def test_conv1dbn_fold_bn():
    torch.manual_seed(42)

    C_IN = 3
    C_OUT = 5
    K = 3
    L = 32
    x = torch.randn(4, C_IN, L)

    conv1d_bn= setup_conv1dBN(C_IN, C_OUT, K, L)
    expected = calc_expected_without_folding(x, conv1d_bn)
    actual = calc_actual_with_folding(x, conv1d_bn)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-6,
    )


def create_ref_model(C_IN, C_OUT, K ,actual_layer):
    ref_conv = nn.Conv1d(
        C_IN,
        C_OUT,
        kernel_size=K,
        padding=1,
        bias=True,
    )
    ref_bn = nn.BatchNorm1d(C_OUT)

    ref_conv.eval()
    ref_bn.eval()
    ref_conv.load_state_dict(actual_layer.conv1d.state_dict())
    ref_bn.load_state_dict(actual_layer.bn1d.state_dict())
    return ref_conv, ref_bn

def create_conv1dBN(C_IN, C_OUT, K, L, BITS, BIAS_BITS,ACTIVATION_BITS):
    layer = Conv1dBN(
        in_channels=C_IN,
        out_channels=C_OUT,
        kernel_size=K,
        seq_len=L,
        padding=1,
        quant_bits=BITS,
        bias_bits=BIAS_BITS,
        activation_bits=ACTIVATION_BITS,
        device="cpu",
    )


    layer.eval()

    with torch.no_grad():
        layer.bn1d.running_mean.copy_(torch.randn(C_OUT))
        layer.bn1d.running_var.copy_(torch.rand(C_OUT) + 0.5)
        layer.bn1d.weight.copy_(torch.randn(C_OUT) + 1.0)
        layer.bn1d.bias.copy_(torch.randn(C_OUT))

    return layer

@pytest.mark.parametrize(
    "B, C_IN, C_OUT, L, K, BITS, BIAS_BITS, ACTIVATION_BITS, seed",
    [
        (4, 3, 5, 32, 3, 8, 8, 8, 42),  # Original configuration
        (1, 1, 1, 16, 3, 8, 8, 8, 43),  # Minimal channels and batch
        (2, 4, 8, 64, 5, 8, 8, 8, 69),  # Larger kernel and sequence
        (8, 3, 6, 32, 3, 4, 8, 8, 100),  # Lower precision
        (4, 8, 16, 128, 3, 8, 8, 8, 2),  # Larger model
        (2, 3, 5, 32, 3, 14, 8, 8, 25),  # Higher precision
        (2, 3, 5, 32, 3, 14, 32, 8, 25),  # 32-bit bias
        (2, 3, 5, 32, 3,8, 32, 16, 25),  # 16-bit activation
    ],
)
def test_conv1dbn_quantized_bn_folding(
    B, C_IN, C_OUT, L, K, BITS, BIAS_BITS,ACTIVATION_BITS, seed
):
    torch.manual_seed(seed)

    actual_layer = create_conv1dBN(C_IN, C_OUT, K, L, BITS, BIAS_BITS, ACTIVATION_BITS)
    ref_conv, ref_bn = create_ref_model(
        C_IN, C_OUT, K, actual_layer
    )

    actual_layer.eval()
    ref_conv.eval()
    ref_bn.eval()

    x = torch.randn(B, C_IN, L)

    with torch.no_grad():
        y_ref = ref_bn(ref_conv(x))
        actual_layer.inputs_QParams.update_quant_params(x)
        y_float = actual_layer(x, enable_simquant=False)
        torch.testing.assert_close(y_float, y_ref, rtol=1e-4, atol=1e-5)

        actual_layer.outputs_QParams.update_quant_params(y_float)

        actual_layer.precompute()

        q_x = actual_layer.inputs_QParams.quantize(x).to(torch.int32)
        q_y = actual_layer.int_forward(q_x)

        y_quantized = actual_layer.outputs_QParams.dequantize(q_y)
        # The chosen bias width is fixed: narrow biases can saturate. Compare
        # integer inference with a reference using those quantized parameters.
        fused_weight, fused_bias = actual_layer._fold_bn()
        input_scale = actual_layer.inputs_QParams.scale_factor
        weight_scale = actual_layer.weight_QParams.scale_factor
        bias_scale = input_scale * weight_scale
        bias_limit = 2 ** (BIAS_BITS - 1) - 1
        reference_bias_codes = (fused_bias / bias_scale).round().to(torch.float64).clamp(
            -bias_limit, bias_limit
        )
        torch.testing.assert_close(
            actual_layer.q_fused_bias,
            reference_bias_codes.to(torch.int32),
        )
        weight_params = actual_layer.weight_QParams
        reference_weight_codes = (fused_weight / weight_scale + weight_params.zero_point).round().clamp(
            weight_params.min_quant.item(), weight_params.max_quant.item()
        ) - weight_params.zero_point
        reference_input = (q_x.to(torch.float64) - actual_layer.inputs_QParams.zero_point) * input_scale
        reference_output = F.conv1d(
            reference_input,
            reference_weight_codes.to(torch.float64) * weight_scale,
            reference_bias_codes * bias_scale,
            padding=actual_layer.padding,
        )
        reference_output = actual_layer.outputs_QParams.dequantize(
            actual_layer.outputs_QParams.quantize(reference_output)
        )
    print("zero point: ", actual_layer.weight_QParams.zero_point)

    error = (y_quantized - reference_output).abs()
    print("Input scale:", actual_layer.inputs_QParams.scale_factor)
    print("Weight scale:", actual_layer.weight_QParams.scale_factor)
    print("Output scale:", actual_layer.outputs_QParams.scale_factor)
    print("Max error:", error.max().item())
    print("Mean error:", error.mean().item())
    tolerance = 3 * actual_layer.outputs_QParams.scale_factor
    print("Max error:", error.max().item())
    print("Mean error:", error.mean().item())
    print("Tolerance:", tolerance)
    assert error.mean() < 0.05

