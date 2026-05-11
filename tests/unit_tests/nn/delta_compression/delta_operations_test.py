"""Unit tests for DeltaOperations class."""

import pytest
import torch

from elasticai.creator.arithmetic.fxp_arithmetic import FxpArithmetic
from elasticai.creator.arithmetic.fxp_params import FxpParams
from elasticai.creator.nn.delta_compression.delta_operations import (
    DeltaOperations,
    DeltaType,
)


class TestDeltaOperationsWithoutClamping:
    """Test cases for DeltaOperations using bitmask class."""

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


class TestDeltaOperationsWithClamping:
    """Test cases for DeltaOperations using clamping class."""

    def test_internal_compress_offset_0_bits_delta_4_bits(self):
        """Test _clamp_lsb clamps values correctly to the full range."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            clamp=True,
        )

        # Test input with values that should be clamped
        input_tensor = torch.Tensor([-16.0, 16.0, -8.0, 8.0, -4.0, 4.0, 0.0])
        result = delta_ops._compress(input_tensor.clone())

        # Verify specific clamping behavior
        # Values outside the range should be clamped to the boundaries
        assert result[0] == -7
        assert result[1] == 7
        assert result[2] == -7
        assert result[3] == 7
        assert result[4] == -4
        assert result[5] == 4
        assert result[6] == 0

    def test_internal_compress_offset_1_bits_delta_4_bits(self):
        """Test _clamp_lsb clamps values correctly to the full range."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=1,
            fxp_arithmetic=fxp_arithmetic,
            clamp=True,
        )

        # Test input with values that should be clamped
        input_tensor = torch.Tensor([-16.0, 16.0, -8.0, 8.0, -4.0, 4.0, 0.0])
        result = delta_ops._compress(input_tensor.clone())

        # Verify specific clamping behavior
        # Values outside the range should be clamped to the boundaries
        assert result[0] == -15
        assert result[1] == 15
        assert result[2] == -8
        assert result[3] == 8
        assert result[4] == -4
        assert result[5] == 4
        assert result[6] == 2

    def test_internal_compress_offset_2_bits_delta_4_bits(self):
        """Test _clamp_lsb clamps values correctly to the full range."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=2,
            fxp_arithmetic=fxp_arithmetic,
            clamp=True,
        )

        # Test input with values that should be clamped
        input_tensor = torch.Tensor([-32.0, 32.0, -16.0, 16.0, -8.0, 8.0])
        result = delta_ops._compress(input_tensor.clone())

        # Verify specific clamping behavior
        # Values outside the range should be clamped to the boundaries
        assert result[0] == -31
        assert result[1] == 31
        assert result[2] == -16
        assert result[3] == 16
        assert result[4] == -8
        assert result[5] == 8

    def test_compress_function(self):
        """Test the compress function with lsb clamp_style."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            clamp=True,
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
            clamp=True,
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
            clamp=True,
        )

        # Test input - use values that work well with fixed-point
        input_tensor = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        compressed = delta_ops.compress(input_tensor.clone())

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


class TestDeltaOperationsFixedReferenceWithoutClamping:
    """Test cases for DeltaOperations using fixed-reference delta encoding."""

    def test_compress_function(self):
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            delta_type=DeltaType.FIXED_REFERENCE,
        )

        # Non-equal spacing makes fixed-reference and consecutive produce different results:
        # FIXED_REFERENCE deltas from element[0]=16: [1, 2, 3, 6]
        # CONSECUTIVE would give:                    [1, 1, 1, 3]
        input_tensor = torch.Tensor([1.0, 1.0625, 1.125, 1.1875, 1.375])
        result = delta_ops.compress(input_tensor.clone())

        assert result.shape == input_tensor.shape
        assert result[0] == fxp_arithmetic.cut_as_integer(1.0)
        assert result[1] == 1
        assert result[2] == 2
        assert result[3] == 3
        assert result[4] == 6

    def test_inflate_function(self):
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            delta_type=DeltaType.FIXED_REFERENCE,
        )

        # First element is the absolute integer value; rest are deltas from it.
        # Fixed-reference inflate adds element[0] to each delta independently,
        # not cumulatively as CONSECUTIVE would.
        input_tensor = torch.Tensor([16.0, 1.0, 2.0, 3.0])
        result = delta_ops.inflate(input_tensor.clone())

        assert result.shape == input_tensor.shape
        assert result[0] == fxp_arithmetic.as_rational(16.0)
        assert result[1] == fxp_arithmetic.as_rational(16.0 + 1.0)
        assert result[2] == fxp_arithmetic.as_rational(16.0 + 2.0)
        assert result[3] == fxp_arithmetic.as_rational(16.0 + 3.0)

    def test_compress_inflate_roundtrip(self):
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            delta_type=DeltaType.FIXED_REFERENCE,
        )

        # Deltas from element[0] are 1, 2, 3, 6 in integer space — all fit in bitmask(4,0)=7
        input_tensor = torch.Tensor([1.0, 1.0625, 1.125, 1.1875, 1.375])

        compressed = delta_ops.compress(input_tensor.clone())
        assert compressed.shape == input_tensor.shape

        inflated = delta_ops.inflate(compressed)
        assert inflated.shape == input_tensor.shape

        assert torch.Tensor.all(inflated == input_tensor)


class TestDeltaOperationsFixedReferenceWithClamping:
    """Test cases for DeltaOperations with fixed-reference delta encoding and clamping."""

    def test_compress_function(self):
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            delta_type=DeltaType.FIXED_REFERENCE,
            clamp=True,
        )

        # Deltas from element[0]=0: 16, -16, 8 all exceed clamp max=7
        input_tensor = torch.Tensor([0.0, 1.0, -1.0, 0.5])
        result = delta_ops.compress(input_tensor.clone())

        assert result.shape == input_tensor.shape
        assert result[0] == fxp_arithmetic.cut_as_integer(0.0)
        assert result[1] == 7
        assert result[2] == -7
        assert result[3] == 7

    def test_inflate_function(self):
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            delta_type=DeltaType.FIXED_REFERENCE,
            clamp=True,
        )

        input_tensor = torch.Tensor([16.0, 1.0, 2.0, 3.0])
        result = delta_ops.inflate(input_tensor.clone())

        assert result.shape == input_tensor.shape
        assert result[0] == fxp_arithmetic.as_rational(16.0)
        assert result[1] == fxp_arithmetic.as_rational(16.0 + 1.0)
        assert result[2] == fxp_arithmetic.as_rational(16.0 + 2.0)
        assert result[3] == fxp_arithmetic.as_rational(16.0 + 3.0)

    def test_compress_inflate_roundtrip(self):
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)

        delta_ops = DeltaOperations(
            delta_bits=4,
            delta_offset=0,
            fxp_arithmetic=fxp_arithmetic,
            delta_type=DeltaType.FIXED_REFERENCE,
            clamp=True,
        )

        # Deltas from element[0] are 1, 2, 3, 6 — all within clamp max=7
        input_tensor = torch.Tensor([1.0, 1.0625, 1.125, 1.1875, 1.375])

        compressed = delta_ops.compress(input_tensor.clone())
        assert compressed.shape == input_tensor.shape

        inflated = delta_ops.inflate(compressed)
        assert inflated.shape == input_tensor.shape

        assert torch.Tensor.all(inflated == input_tensor)


class TestDeltaOperationsPostInit:
    """Test that invalid constructor arguments raise ValueError."""

    def test_raises_on_zero_delta_bits(self):
        fxp_arithmetic = FxpArithmetic(fxp_params=FxpParams(total_bits=8, frac_bits=4))
        with pytest.raises(ValueError):
            DeltaOperations(delta_bits=0, delta_offset=0, fxp_arithmetic=fxp_arithmetic)

    def test_raises_on_negative_delta_bits(self):
        fxp_arithmetic = FxpArithmetic(fxp_params=FxpParams(total_bits=8, frac_bits=4))
        with pytest.raises(ValueError):
            DeltaOperations(delta_bits=-1, delta_offset=0, fxp_arithmetic=fxp_arithmetic)

    def test_raises_on_negative_delta_offset(self):
        fxp_arithmetic = FxpArithmetic(fxp_params=FxpParams(total_bits=8, frac_bits=4))
        with pytest.raises(ValueError):
            DeltaOperations(delta_bits=4, delta_offset=-1, fxp_arithmetic=fxp_arithmetic)

    def test_raises_when_delta_bits_plus_offset_exceeds_total_bits(self):
        fxp_arithmetic = FxpArithmetic(fxp_params=FxpParams(total_bits=8, frac_bits=4))
        with pytest.raises(ValueError):
            DeltaOperations(delta_bits=5, delta_offset=4, fxp_arithmetic=fxp_arithmetic)

    def test_valid_when_delta_bits_plus_offset_equals_total_bits(self):
        """delta_bits + delta_offset == total_bits is the exact boundary and must be accepted."""
        fxp_arithmetic = FxpArithmetic(fxp_params=FxpParams(total_bits=8, frac_bits=4))
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=4, fxp_arithmetic=fxp_arithmetic
        )
        assert delta_ops.delta_bits == 4
        assert delta_ops.delta_offset == 4


class TestDeltaOperationsEdgeCases:
    """Boundary and edge-case inputs for DeltaOperations."""

    def test_single_element_compress(self):
        """A length-1 tensor has no deltas; the sole element is returned unchanged as an integer."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        result = delta_ops.compress(torch.Tensor([1.5]))
        assert result.shape == torch.Size([1])
        assert result[0] == fxp_arithmetic.cut_as_integer(1.5)

    def test_single_element_inflate(self):
        """A length-1 tensor has no deltas to reverse; the element is converted to rational."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        result = delta_ops.inflate(torch.Tensor([16.0]))
        assert result.shape == torch.Size([1])
        assert result[0] == fxp_arithmetic.as_rational(16.0)

    def test_multidimensional_compress_preserves_shape(self):
        """compress flattens internally but must restore the original tensor shape."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        input_tensor = torch.Tensor([[1.0, 1.0625], [1.125, 1.1875]])
        result = delta_ops.compress(input_tensor.clone())
        assert result.shape == torch.Size([2, 2])

    def test_multidimensional_inflate_preserves_shape(self):
        """inflate flattens internally but must restore the original tensor shape."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        input_tensor = torch.Tensor([[16.0, 1.0], [1.0, 1.0]])
        result = delta_ops.inflate(input_tensor.clone())
        assert result.shape == torch.Size([2, 2])

    def test_all_identical_values_produce_zero_deltas(self):
        """When all values are equal, all consecutive deltas are zero after compression."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        result = delta_ops.compress(torch.Tensor([1.0, 1.0, 1.0, 1.0]))
        assert result[0] == fxp_arithmetic.cut_as_integer(1.0)
        assert torch.all(result[1:] == 0)

    def test_consecutive_compress_with_negative_deltas(self):
        """Descending values produce negative deltas; compress must preserve their sign."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=4, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        # Each step decreases by 1 LSB in integer space (frac_bits=4, so 0.0625 = 1 unit)
        # cut_as_integer: [4, 3, 2, 1] → consecutive deltas: [4, -1, -1, -1]
        result = delta_ops.compress(torch.Tensor([0.25, 0.1875, 0.125, 0.0625]))
        assert result[0] == fxp_arithmetic.cut_as_integer(0.25)
        assert result[1] == -1
        assert result[2] == -1
        assert result[3] == -1

    def test_delta_bits_one_without_clamping_zeros_all_deltas(self):
        """With delta_bits=1 the bitmask is 0 (empty loop in __bitmask), collapsing every delta to zero."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=1, delta_offset=0, fxp_arithmetic=fxp_arithmetic
        )
        result = delta_ops.compress(torch.Tensor([1.0, 2.0, 3.0]))
        assert result[0] == fxp_arithmetic.cut_as_integer(1.0)
        assert torch.all(result[1:] == 0)

    def test_delta_bits_one_with_clamping_and_nonzero_offset_silently_clamps_to_max(self):
        """With delta_bits=1 and delta_offset>0 the clamp window is inverted (min=2 > max=1).
        PyTorch silently clamps to max, so every delta becomes 1 regardless of its actual value."""
        fxp_params = FxpParams(total_bits=8, frac_bits=4)
        fxp_arithmetic = FxpArithmetic(fxp_params=fxp_params)
        delta_ops = DeltaOperations(
            delta_bits=1, delta_offset=1, fxp_arithmetic=fxp_arithmetic, clamp=True
        )
        result = delta_ops.compress(torch.Tensor([1.0, 2.0, 3.0]))
        assert result[0] == fxp_arithmetic.cut_as_integer(1.0)
        assert torch.all(result[1:] == 1)
