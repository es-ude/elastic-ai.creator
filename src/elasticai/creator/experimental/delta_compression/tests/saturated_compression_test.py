import pytest
import torch

from elasticai.creator.experimental.delta_compression.src.compression.saturate import (
    SaturatedCompression,
)

# SaturatedCompression(width, offset):
#   compress range: abs clamped to [2^offset (or 0 if offset==0), 2^(width-1+offset)-1]
#   sign preserved; inflate is identity


class TestCompressRange:
    def test_value_within_range_unchanged(self):
        # width=4, offset=0 → range [0, 7]
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([3.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(3.0)

    def test_value_above_max_clamped_to_max(self):
        # max = 2^(4-1+0)-1 = 7
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([100.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(7.0)

    def test_value_at_max_boundary_unchanged(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([7.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(7.0)

    def test_zero_with_zero_offset_unchanged(self):
        # min=0 when offset==0, so zero is valid
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([0.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(0.0)

    def test_value_below_min_clamped_up_when_offset_nonzero(self):
        # width=4, offset=2 → min=2^2=4, max=2^(4-1+2)-1=31
        sc = SaturatedCompression(width=4, offset=2)
        x = torch.tensor([1.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(4.0)

    def test_value_above_max_with_nonzero_offset(self):
        # width=4, offset=2 → max=31
        sc = SaturatedCompression(width=4, offset=2)
        x = torch.tensor([100.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(31.0)

    def test_width_one_offset_zero_clamps_all_to_zero(self):
        # min=0, max=2^(1-1+0)-1=0 → everything becomes 0
        sc = SaturatedCompression(width=1, offset=0)
        x = torch.tensor([5.0, -3.0, 0.0])
        result = sc.compress(x, in_place=False)
        assert result.tolist() == pytest.approx([0.0, 0.0, 0.0])


class TestCompressSign:
    def test_negative_value_within_range_preserved(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([-5.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(-5.0)

    def test_negative_value_above_abs_max_clamped_with_sign(self):
        # abs(-100)=100 > 7 → clamped to 7 → restored to -7
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([-100.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(-7.0)

    def test_negative_value_below_min_clamped_with_sign(self):
        # width=4, offset=2 → min=4; abs(-1)=1 < 4 → clamped to 4 → -4
        sc = SaturatedCompression(width=4, offset=2)
        x = torch.tensor([-1.0])
        assert sc.compress(x, in_place=False).item() == pytest.approx(-4.0)

    def test_mixed_signs_handled_independently(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([3.0, -3.0, 10.0, -10.0])
        result = sc.compress(x, in_place=False)
        assert result.tolist() == pytest.approx([3.0, -3.0, 7.0, -7.0])


class TestCompressShape:
    def test_1d_shape_preserved(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([1.0, 2.0, 3.0])
        assert sc.compress(x, in_place=False).shape == x.shape

    def test_2d_shape_preserved(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([[1.0, 10.0], [-2.0, -20.0]])
        assert sc.compress(x, in_place=False).shape == x.shape

    def test_2d_values_correct_after_reshape(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([[1.0, 10.0], [-2.0, -20.0]])
        result = sc.compress(x, in_place=False)
        assert result[0].tolist() == pytest.approx([1.0, 7.0])
        assert result[1].tolist() == pytest.approx([-2.0, -7.0])


class TestInPlace:
    def test_in_place_false_does_not_modify_input(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([1.0, 100.0, -50.0])
        original = x.clone()
        sc.compress(x, in_place=False)
        assert torch.equal(x, original)

    def test_in_place_true_modifies_input(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([100.0])
        sc.compress(x, in_place=True)
        assert x.item() == pytest.approx(7.0)


class TestInflate:
    def test_inflate_is_identity(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([1.0, -3.0, 7.0])
        result = sc.inflate(x, in_place=False)
        assert torch.equal(result, x)

    def test_inflate_in_place_false_does_not_modify_input(self):
        sc = SaturatedCompression(width=4, offset=0)
        x = torch.tensor([1.0, 2.0])
        original = x.clone()
        sc.inflate(x, in_place=False)
        assert torch.equal(x, original)
