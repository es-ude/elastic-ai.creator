"""Unit tests for DeltaOperations class."""

import torch

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic
from elasticai.creator.arithmetic.fxp_params import FxpParams
from elasticai.creator.nn.delta_compression.delta_operations import DeltaOperations


class TestDeltaOperations:
    """Test cases for DeltaOperations class."""

    def test_internal_compress_offset_0_bits_delta_4_bits(self):
        """Test _clamp_lsb clamps values correctly to the full range."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
        )

        # Test input with values that should be clamped
        input_tensor = torch.Tensor([-9.0, 9.0, -4.0, 4.0, -2.0, 1.0])
        result = delta_ops._compress(input_tensor.clone())

        # Verify the result is clamped to the delta range
        min_val = -(2 ** (delta_ops.delta_bits - 1)) - 1
        max_val = 2 ** (delta_ops.delta_bits - 1)

        # Verify all values are within the clamped range
        assert torch.Tensor.all(result <= max_val)
        assert torch.Tensor.all(result >= min_val)

        # Verify specific clamping behavior
        # Values outside the range should be clamped to the boundaries
        assert result[0] == -1
        assert result[1] == 1
        assert result[2] == -4
        assert result[3] == 4
        assert result[4] == -2
        assert result[5] == 1

    def test_internal_compress_offset_4_bits_delta_4_bits(self):
        """Test _clamp_lsb clamps values correctly to the full range."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=4,
            fxp_arithmetic=fxp_arithmetic,
        )

        # Test input with values that should be clamped
        input_tensor = torch.Tensor(
            [-72.0, 72.0, -64.0, 64.0, -136.0, 136.0, -144, 144]
        )
        result = delta_ops._compress(input_tensor.clone())

        # Verify the result is clamped to the delta range
        min_val = -(2 ** (delta_ops.delta_bits + delta_ops.delta_offset - 1)) - 1
        max_val = 2 ** (delta_ops.delta_bits + delta_ops.delta_offset - 1)

        # Verify all values are within the clamped range
        assert torch.Tensor.all(result <= max_val)
        assert torch.Tensor.all(result >= min_val)

        # Verify specific clamping behavior
        # Values outside the range should be clamped to the boundaries
        assert result[0] == -64
        assert result[1] == 64
        assert result[2] == -64
        assert result[3] == 64
        assert result[4] == 0
        assert result[5] == 0
        assert result[6] == -16
        assert result[7] == 16

    def test_compress_function(self):
        """Test the compress function with lsb clamp_style."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
        )

        # Test input - use values that are within the fixed-point range
        input_tensor = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        input_shape = input_tensor.shape
        result = delta_ops.compress(input_tensor)

        # Verify the result shape is preserved
        assert result.shape == input_shape

        # Verify first element not changed
        assert result[0] == fxp_arithmetic.cut_as_integer(1.0)

        # But clamped to delta_bits range
        min_val = -(2 ** (delta_ops.delta_bits - 1))
        max_val = 2 ** (delta_ops.delta_bits - 1)
        assert torch.Tensor.all(result >= min_val)
        assert torch.Tensor.all(result[1:] <= max_val)

    def test_inflate_function(self):
        """Test the inflate function with lsb clamp_style."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
        )

        input_tensor = torch.Tensor([4.0, -4.0, 0.0, 2.0])
        result = delta_ops.inflate(input_tensor.clone())

        # Verify the result shape is preserved
        assert result.shape == input_tensor.shape

        # Verify the inflated values are reasonable (within fixed-point range)
        min_val = -(2 ** (delta_ops.total_bits - 1)) - 1
        max_val = 2 ** (delta_ops.total_bits - 1)
        assert torch.Tensor.all(result >= min_val)
        assert torch.Tensor.all(result <= max_val)

        # Verify first element is not changed
        assert result[0] == fxp_arithmetic.as_rational(input_tensor[0])

        # Verify the inflation produces a cumulative sum pattern
        assert result[1] == fxp_arithmetic.as_rational(
            input_tensor[0] + input_tensor[1]
        )
        assert result[2] == fxp_arithmetic.as_rational(
            input_tensor[0] + input_tensor[1] + input_tensor[2]
        )
        assert result[3] == fxp_arithmetic.as_rational(
            input_tensor[0] + input_tensor[1] + input_tensor[2] + input_tensor[3]
        )

    def test_compress_inflate_roundtrip(self):
        """Test that compress followed by inflate produces reasonable results with lsb clamp_style."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=2,
            fxp_arithmetic=fxp_arithmetic,
        )

        # Test input - use values that work well with fixed-point
        input_tensor = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        compressed = delta_ops.compress(input_tensor.clone())

        print(compressed)
        # Verify shapes are preserved
        assert compressed.shape == input_tensor.shape

        inflated = delta_ops.inflate(compressed)

        # Verify shapes are preserved
        assert inflated.shape == input_tensor.shape

        # Verify the inflated values are reasonable (within fixed-point range)
        min_val = -(2 ** (delta_ops.total_bits - 1)) - 1
        max_val = 2 ** (delta_ops.total_bits - 1)
        assert torch.Tensor.all(inflated >= min_val)
        assert torch.Tensor.all(inflated <= max_val)

        # Verify that the inflated values follow the expected pattern
        for i in range(len(inflated) - 1):
            assert inflated[i] <= inflated[i + 1]

        assert torch.Tensor.all(inflated == input_tensor)
