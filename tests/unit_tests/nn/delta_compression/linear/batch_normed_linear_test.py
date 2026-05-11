import pytest
import torch

from elasticai.creator.nn.delta_compression import DeltaType
from elasticai.creator.nn.delta_compression.linear.layer import BatchNormedLinear


def _make_layer(
    in_features: int = 3,
    out_features: int = 2,
    total_bits: int = 8,
    frac_bits: int = 4,
    delta_bits: int = 4,
    delta_offset: int = 0,
    delta_type: DeltaType = DeltaType.CONSECUTIVE,
    clamp: bool = False,
    bias: bool = True,
    bn_eps: float = 1e-5,
    bn_affine: bool = True,
) -> BatchNormedLinear:
    return BatchNormedLinear(
        in_features=in_features,
        out_features=out_features,
        total_bits=total_bits,
        frac_bits=frac_bits,
        delta_bits=delta_bits,
        delta_offset=delta_offset,
        delta_type=delta_type,
        clamp=clamp,
        bias=bias,
        bn_eps=bn_eps,
        bn_affine=bn_affine,
    )


class TestProperties:
    def test_lin_weight_returns_linear_weight(self):
        layer = _make_layer()
        assert layer.lin_weight is layer._linear.weight

    def test_lin_bias_returns_linear_bias(self):
        layer = _make_layer()
        assert layer.lin_bias is layer._linear.bias

    def test_bn_weight_returns_batch_norm_weight(self):
        layer = _make_layer()
        assert layer.bn_weight is layer._batch_norm.weight

    def test_bn_bias_returns_batch_norm_bias(self):
        layer = _make_layer()
        assert layer.bn_bias is layer._batch_norm.bias


class TestForward:
    def test_output_shape_for_batched_input(self):
        layer = _make_layer(in_features=3, out_features=2)
        layer.eval()
        assert layer(torch.zeros(4, 3)).shape == (4, 2)

    def test_output_shape_for_unbatched_input(self):
        """A 1D (unbatched) input must produce a 1D output, not shape (1, out_features)."""
        layer = _make_layer(in_features=3, out_features=2)
        layer.eval()
        assert layer(torch.zeros(3)).shape == (2,)

    def test_forward_without_bias(self):
        layer = _make_layer(in_features=3, out_features=2, bias=False)
        layer.eval()
        assert layer(torch.zeros(4, 3)).shape == (4, 2)

    def test_output_is_on_fxp_grid(self):
        """forward quantizes the output to the fxp grid (multiples of 1/2^frac_bits)."""
        frac_bits = 4
        layer = _make_layer(in_features=2, out_features=1, frac_bits=frac_bits)
        layer.eval()
        y = layer(torch.zeros(4, 2))
        scaled = y / (1.0 / 2**frac_bits)
        assert torch.allclose(scaled, scaled.round(), atol=0.01)


class TestGetParams:
    def test_get_params_shapes(self):
        layer = _make_layer(in_features=3, out_features=2)
        weights, bias = layer.get_params()
        assert len(weights) == 2  # out_features rows
        assert len(weights[0]) == 3  # in_features cols
        assert len(bias) == 2  # one entry per output neuron

    def test_get_params_fuses_batchnorm_without_affine(self):
        """Weights are divided by BN std; bias is shifted by BN mean and divided by std."""
        # bn_eps=1.0 + running_var=3.0 → std = sqrt(4.0) = 2.0 exactly
        layer = _make_layer(in_features=2, out_features=1, bn_eps=1.0, bn_affine=False)
        layer._linear.weight.data = torch.tensor([[2.0, 4.0]])
        layer._linear.bias.data = torch.tensor([1.0])
        layer._batch_norm.running_var.data = torch.tensor([3.0])
        layer._batch_norm.running_mean.data = torch.tensor([0.5])

        weights, bias = layer.get_params()

        assert weights[0] == pytest.approx([1.0, 2.0])  # [2, 4] / 2.0
        assert bias == pytest.approx([0.25])  # (1.0 - 0.5) / 2.0

    def test_get_params_fuses_batchnorm_with_affine(self):
        """With BN affine params, weights/bias are additionally scaled by gamma and shifted by beta."""
        layer = _make_layer(in_features=2, out_features=1, bn_eps=1.0, bn_affine=True)
        layer._linear.weight.data = torch.tensor([[2.0, 4.0]])
        layer._linear.bias.data = torch.tensor([1.0])
        layer._batch_norm.running_var.data = torch.tensor([3.0])  # std = 2.0
        layer._batch_norm.running_mean.data = torch.tensor([0.5])
        layer._batch_norm.weight.data = torch.tensor([3.0])  # gamma
        layer._batch_norm.bias.data = torch.tensor([0.5])  # beta

        weights, bias = layer.get_params()

        # gamma * (lin_weight / std) = 3 * [1.0, 2.0] = [3.0, 6.0]
        assert weights[0] == pytest.approx([3.0, 6.0])
        # gamma * (lin_bias - mean) / std + beta = 3.0 * 0.25 + 0.5 = 1.25
        assert bias == pytest.approx([1.25])

    def test_get_params_uses_zero_bias_when_bias_is_false(self):
        """With bias=False the fused bias comes entirely from -BN_mean/std."""
        layer = _make_layer(
            in_features=2, out_features=1, bias=False, bn_eps=1.0, bn_affine=False
        )
        layer._linear.weight.data = torch.tensor([[1.0, 1.0]])
        layer._batch_norm.running_var.data = torch.tensor([3.0])
        layer._batch_norm.running_mean.data = torch.tensor([0.0])

        _, bias = layer.get_params()

        assert bias == pytest.approx([0.0])  # (0.0 - 0.0) / 2.0 = 0.0


class TestGetParamsQuant:
    def test_get_params_quant_shapes_match_get_params(self):
        layer = _make_layer(in_features=3, out_features=2)
        weights, bias = layer.get_params()
        q_weights, q_bias = layer.get_params_quant()
        assert len(q_weights) == len(weights)
        assert len(q_weights[0]) == len(weights[0])
        assert len(q_bias) == len(bias)

    def test_get_params_quant_known_value(self):
        """Fused weight 1.0 with frac_bits=4 maps to integer 16 (= 1.0 / step)."""
        layer = _make_layer(
            in_features=2, out_features=1, frac_bits=4, bn_eps=1.0, bn_affine=False
        )
        layer._linear.weight.data = torch.tensor([[2.0, 4.0]])
        layer._linear.bias.data = torch.tensor([0.0])
        layer._batch_norm.running_var.data = torch.tensor([3.0])  # std = 2.0
        layer._batch_norm.running_mean.data = torch.tensor([0.0])
        # fused weight[0][0] = 2.0 / 2.0 = 1.0 → cut_as_integer = 1.0 / (1/16) = 16

        q_weights, _ = layer.get_params_quant()

        assert q_weights[0][0] == 16


class TestGetParamsCompressed:
    def test_get_params_compressed_shapes_match_get_params(self):
        layer = _make_layer(in_features=3, out_features=2)
        weights, bias = layer.get_params()
        c_weights, c_bias = layer.get_params_compressed()
        assert len(c_weights) == len(weights)
        assert len(c_weights[0]) == len(weights[0])
        assert len(c_bias) == len(bias)

    def test_get_params_compressed_first_weight_is_absolute_integer(self):
        """First element of a compressed row is the absolute fxp integer, not a delta."""
        layer = _make_layer(
            in_features=2, out_features=1, frac_bits=4, bn_eps=1.0, bn_affine=False
        )
        layer._linear.weight.data = torch.tensor([[2.0, 4.0]])
        layer._linear.bias.data = torch.tensor([0.0])
        layer._batch_norm.running_var.data = torch.tensor([3.0])  # std = 2.0
        layer._batch_norm.running_mean.data = torch.tensor([0.0])
        # fused weight[0][0] = 1.0 → integer 16

        c_weights, _ = layer.get_params_compressed()

        assert c_weights[0][0] == 16.0

    def test_get_params_compressed_subsequent_weights_are_deltas(self):
        """Weights after the first are stored as consecutive differences in integer space."""
        layer = _make_layer(
            in_features=2, out_features=1, frac_bits=4, bn_eps=1.0, bn_affine=False
        )
        # fused: [[1.0, 1.0625]]; integers: [16, 17]; delta = 1
        layer._linear.weight.data = torch.tensor([[2.0, 2.125]])
        layer._linear.bias.data = torch.tensor([0.0])
        layer._batch_norm.running_var.data = torch.tensor([3.0])  # std = 2.0
        layer._batch_norm.running_mean.data = torch.tensor([0.0])

        c_weights, _ = layer.get_params_compressed()

        assert c_weights[0][0] == 16.0  # absolute integer
        assert c_weights[0][1] == 1.0  # delta = 17 - 16 = 1

    def test_get_params_compressed_bias_first_element_is_absolute_integer(self):
        """Bias (single feature) is stored as its absolute fxp integer with no delta."""
        layer = _make_layer(
            in_features=2, out_features=1, frac_bits=4, bn_eps=1.0, bn_affine=False
        )
        layer._linear.weight.data = torch.tensor([[1.0, 1.0]])
        layer._linear.bias.data = torch.tensor([1.0])
        layer._batch_norm.running_var.data = torch.tensor([3.0])  # std = 2.0
        layer._batch_norm.running_mean.data = torch.tensor([0.0])
        # fused bias = 1.0 / 2.0 = 0.5 → integer 8

        _, c_bias = layer.get_params_compressed()

        assert c_bias[0] == 8.0


@pytest.mark.slow
def test_training_loop_completes_without_error():
    layer = _make_layer(in_features=4, out_features=4)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(layer.parameters())
    x = torch.rand(8, 4)
    for _ in range(3):
        y = layer(x)
        loss = criterion(y, torch.zeros_like(y))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class TestCreateDesign:
    def test_create_design_raises_not_implemented(self):
        layer = _make_layer()
        with pytest.raises(NotImplementedError):
            layer.create_design("test_design")
