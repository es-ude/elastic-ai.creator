import torch

from elasticai.creator.experimental.delta_compression.src.compression.bitmask import (
    BitMaskCompression,
)

# BitMaskCompression(width, offset):
#   bitmask = OR of bits [offset, offset+1, ..., offset+width-2]  (range(width-1))
#   compress: abs value → bitwise AND with bitmask → restore sign
#   inflate: identity
#
# Bitmask examples:
#   width=4, offset=0 → bits 0,1,2 → mask=0b0111=7
#   width=4, offset=2 → bits 2,3,4 → mask=0b11100=28
#   width=1, offset=0 → range(0) empty → mask=0
#   width=2, offset=0 → bit 0 → mask=1
#   width=3, offset=1 → bits 1,2 → mask=0b110=6
#
# Input must be integer tensor (&= not defined for float).


def t(*values: int) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int32)


class TestBitmaskAndRange:
    def test_value_within_mask_bits_unchanged(self):
        # mask=7 (bits 0-2); 5=0b101 → 5&7=5
        cc = BitMaskCompression(width=4, offset=0)
        assert cc.compress(t(5), in_place=False).item() == 5

    def test_value_with_bit_outside_mask_zeroed(self):
        # mask=7; 8=0b1000 → 8&7=0
        cc = BitMaskCompression(width=4, offset=0)
        assert cc.compress(t(8), in_place=False).item() == 0

    def test_value_partially_masked(self):
        # mask=7; 15=0b1111 → 15&7=7
        cc = BitMaskCompression(width=4, offset=0)
        assert cc.compress(t(15), in_place=False).item() == 7

    def test_nonzero_offset_shifts_bitmask(self):
        # width=4, offset=2 → mask=28=0b11100; 5=0b00101 → 5&28=4
        cc = BitMaskCompression(width=4, offset=2)
        assert cc.compress(t(5), in_place=False).item() == 4

    def test_nonzero_offset_value_within_mask_unchanged(self):
        # mask=28; 12=0b01100 → 12&28=12
        cc = BitMaskCompression(width=4, offset=2)
        assert cc.compress(t(12), in_place=False).item() == 12

    def test_width_one_zeros_all_values(self):
        # range(width-1)=range(0) → mask=0 → everything zeroed
        cc = BitMaskCompression(width=1, offset=0)
        result = cc.compress(t(1, 5, 100), in_place=False)
        assert result.tolist() == [0, 0, 0]

    def test_width_two_offset_zero_keeps_only_lsb(self):
        # mask=1; 3=0b11 → 1; 4=0b100 → 0; 5=0b101 → 1
        cc = BitMaskCompression(width=2, offset=0)
        result = cc.compress(t(3, 4, 5), in_place=False)
        assert result.tolist() == [1, 0, 1]

    def test_width_three_offset_one_applies_correct_mask(self):
        # bits 1,2 → mask=0b110=6; 7=0b111 → 7&6=6; 5=0b101 → 5&6=4
        cc = BitMaskCompression(width=3, offset=1)
        result = cc.compress(t(7, 5), in_place=False)
        assert result.tolist() == [6, 4]


class TestSign:
    def test_negative_value_within_mask_preserved(self):
        # abs(-5)=5, 5&7=5, restore sign → -5
        cc = BitMaskCompression(width=4, offset=0)
        assert cc.compress(t(-5), in_place=False).item() == -5

    def test_negative_value_with_bit_outside_mask_becomes_zero(self):
        # abs(-8)=8, 8&7=0, restore sign → 0
        cc = BitMaskCompression(width=4, offset=0)
        assert cc.compress(t(-8), in_place=False).item() == 0

    def test_negative_partially_masked_sign_restored(self):
        # abs(-15)=15, 15&7=7, restore sign → -7
        cc = BitMaskCompression(width=4, offset=0)
        assert cc.compress(t(-15), in_place=False).item() == -7

    def test_mixed_positive_negative(self):
        # mask=7; [3,-3,8,-8] → [3,-3,0,0]
        cc = BitMaskCompression(width=4, offset=0)
        result = cc.compress(t(3, -3, 8, -8), in_place=False)
        assert result.tolist() == [3, -3, 0, 0]


class TestShape:
    def test_1d_shape_preserved(self):
        cc = BitMaskCompression(width=4, offset=0)
        x = t(1, 2, 3)
        assert cc.compress(x, in_place=False).shape == x.shape

    def test_2d_shape_preserved(self):
        cc = BitMaskCompression(width=4, offset=0)
        x = torch.tensor([[1, 8], [3, 15]], dtype=torch.int32)
        assert cc.compress(x, in_place=False).shape == x.shape

    def test_2d_values_correct_after_reshape(self):
        # mask=7; [[1,8],[3,15]] → [[1,0],[3,7]]
        cc = BitMaskCompression(width=4, offset=0)
        x = torch.tensor([[1, 8], [3, 15]], dtype=torch.int32)
        result = cc.compress(x, in_place=False)
        assert result[0].tolist() == [1, 0]
        assert result[1].tolist() == [3, 7]


class TestInPlace:
    def test_in_place_false_does_not_modify_input(self):
        cc = BitMaskCompression(width=4, offset=0)
        x = t(1, 8, 15)
        original = x.clone()
        cc.compress(x, in_place=False)
        assert torch.equal(x, original)

    def test_in_place_true_modifies_input(self):
        cc = BitMaskCompression(width=4, offset=0)
        x = t(15)
        cc.compress(x, in_place=True)
        assert x.item() == 7


class TestInflate:
    def test_inflate_is_identity(self):
        cc = BitMaskCompression(width=4, offset=0)
        x = t(1, 3, 7)
        result = cc.inflate(x, in_place=False)
        assert torch.equal(result, x)

    def test_inflate_in_place_false_does_not_modify_input(self):
        cc = BitMaskCompression(width=4, offset=0)
        x = t(1, 2, 3)
        original = x.clone()
        cc.inflate(x, in_place=False)
        assert torch.equal(x, original)
