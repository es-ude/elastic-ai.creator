import pytest
import torch

from elasticai.creator.experimental.delta_compression.src.builder import (
    DeltaCompBuilder,
)
from elasticai.creator.experimental.delta_compression.src.delta_decorator import (
    delta_compressed_bias,
    delta_compressed_weights,
)
from elasticai.creator.nn.fixed_point.linear.layer import Linear


def _make_dc():
    return (
        DeltaCompBuilder()
        .consecutive_delta()
        .saturated_compression(delta_width=4, offset=0)
        .build()
    )


class TestDeltaCompressedWeights:
    def test_decorator_replaces_weight_with_descriptor(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        assert hasattr(layer, "_original_weight")
        assert isinstance(layer._original_weight, torch.nn.Parameter)

    def test_weight_access_returns_compressed(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        weight = layer.weight
        # Weight should be different from original due to compression
        # (unless original is already in compressed form)
        assert weight.shape == layer._original_weight.shape

    def test_forward_uses_compressed_weights(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=2, out_features=1, total_bits=8, frac_bits=4
        )
        # Set weights that will be affected by compression
        layer._original_weight.data = torch.tensor([[0.0, 1000.0]])

        x = torch.ones(1, 2)
        y = layer(x)

        # Due to compression, output should differ from uncompressed
        uncompressed_layer = Linear(
            in_features=2, out_features=1, total_bits=8, frac_bits=4
        )
        uncompressed_layer.weight.data = torch.tensor([[0.0, 1000.0]])
        uncompressed_y = uncompressed_layer(x)

        assert not torch.allclose(y, uncompressed_y)

    def test_gradients_propagate_through_compressed_weights(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        x = torch.randn(4, 4)
        y = layer(x)
        y.sum().backward()

        assert layer._original_weight.grad is not None

    def test_requires_operations_attribute(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        class RegularLinear(torch.nn.Linear):
            pass

        with pytest.raises(TypeError, match="_operations"):
            RegularLinear(4, 2)

    def test_preserves_original_weight_on_set(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        original_weight = layer._original_weight.clone()

        # Set new weight via data attribute
        new_weight = torch.randn_like(layer._original_weight)
        layer._original_weight.data = new_weight

        # Original should be updated
        assert not torch.equal(layer._original_weight, original_weight)


class TestDeltaCompressedBias:
    def test_decorator_replaces_bias_with_descriptor(self):
        dc = _make_dc()

        @delta_compressed_bias(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        assert hasattr(layer, "_original_bias")
        assert isinstance(layer._original_bias, torch.nn.Parameter)

    def test_bias_access_returns_compressed(self):
        dc = _make_dc()

        @delta_compressed_bias(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        bias = layer.bias
        assert bias.shape == layer._original_bias.shape

    def test_forward_uses_compressed_bias(self):
        dc = _make_dc()

        @delta_compressed_bias(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=2, out_features=1, total_bits=8, frac_bits=4
        )

        x = torch.ones(1, 2)
        y = layer(x)

        # Verify forward pass works
        assert y.shape == (1, 1)
        # Verify gradients flow
        y.sum().backward()
        assert layer._original_bias.grad is not None

    def test_gradients_propagate_through_compressed_bias(self):
        dc = _make_dc()

        @delta_compressed_bias(dc)
        class CompressedLinear(Linear):
            pass

        layer = CompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        x = torch.randn(4, 4)
        y = layer(x)
        y.sum().backward()

        assert layer._original_bias.grad is not None

    def test_requires_operations_attribute(self):
        dc = _make_dc()

        @delta_compressed_bias(dc)
        class RegularLinear(torch.nn.Linear):
            pass

        with pytest.raises(TypeError, match="_operations"):
            RegularLinear(4, 2)


class TestCombinedDecorators:
    def test_both_decorators_on_same_layer(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        @delta_compressed_bias(dc)
        class FullyCompressedLinear(Linear):
            pass

        layer = FullyCompressedLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        assert hasattr(layer, "_original_weight")
        assert hasattr(layer, "_original_bias")

    def test_both_parameters_compressed_in_forward(self):
        dc = _make_dc()

        @delta_compressed_weights(dc)
        @delta_compressed_bias(dc)
        class FullyCompressedLinear(Linear):
            pass

        layer = FullyCompressedLinear(
            in_features=2, out_features=1, total_bits=8, frac_bits=4
        )

        x = torch.ones(1, 2)
        y = layer(x)

        # Verify forward pass works and produces output
        assert y.shape == (1, 1)
        # Verify gradients flow
        y.sum().backward()
        assert layer._original_weight.grad is not None
        assert layer._original_bias.grad is not None

    def test_different_compression_for_weights_and_bias(self):
        dc_weights = (
            DeltaCompBuilder()
            .consecutive_delta()
            .saturated_compression(delta_width=4, offset=0)
            .build()
        )
        dc_bias = (
            DeltaCompBuilder()
            .consecutive_delta()
            .saturated_compression(delta_width=2, offset=0)
            .build()
        )

        @delta_compressed_weights(dc_weights)
        @delta_compressed_bias(dc_bias)
        class MixedCompressionLinear(Linear):
            pass

        layer = MixedCompressionLinear(
            in_features=4, out_features=2, total_bits=8, frac_bits=4
        )
        assert hasattr(layer, "_original_weight")
        assert hasattr(layer, "_original_bias")
