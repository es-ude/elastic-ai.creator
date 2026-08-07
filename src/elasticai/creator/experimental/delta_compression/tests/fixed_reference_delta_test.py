import pytest
import torch

from elasticai.creator.experimental.delta_compression.src.delta.fixed_reference_delta import (
    FixedReferenceDelta,
)

# FixedReferenceDelta:
#   delta:       first element unchanged; rest become input[i] - input[0]
#   reconstruct: first element unchanged; rest become input[i] + input[0]
#   Both flatten internally and reshape to original shape.
#
# Contrast with ConsecutiveDelta where each element subtracts its predecessor;
# here every element subtracts the same fixed reference (the first element).


class TestDelta:
    def test_single_element_unchanged(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([7.0])
        assert frd.delta(x, in_place=False).item() == pytest.approx(7.0)

    def test_first_element_always_unchanged(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([5.0, 1.0, 2.0])
        assert frd.delta(x, in_place=False)[0].item() == pytest.approx(5.0)

    def test_increasing_sequence(self):
        # first=1; rest: 2-1=1, 3-1=2, 4-1=3
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert frd.delta(x, in_place=False).tolist() == pytest.approx(
            [1.0, 1.0, 2.0, 3.0]
        )

    def test_decreasing_sequence(self):
        # first=4; rest: 3-4=-1, 2-4=-2, 1-4=-3
        frd = FixedReferenceDelta()
        x = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert frd.delta(x, in_place=False).tolist() == pytest.approx(
            [4.0, -1.0, -2.0, -3.0]
        )

    def test_equal_values_produce_zero_deltas(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([3.0, 3.0, 3.0, 3.0])
        assert frd.delta(x, in_place=False).tolist() == pytest.approx(
            [3.0, 0.0, 0.0, 0.0]
        )

    def test_all_subtract_first_not_predecessor(self):
        # key difference from ConsecutiveDelta: each element subtracts input[0], not input[i-1]
        # [1, 3, 6] → consecutive: [1,2,3]; fixed-ref: [1, 3-1=2, 6-1=5]
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 3.0, 6.0])
        assert frd.delta(x, in_place=False).tolist() == pytest.approx([1.0, 2.0, 5.0])

    def test_mixed_signs(self):
        # first=-2; rest: 1-(-2)=3, -1-(-2)=1
        frd = FixedReferenceDelta()
        x = torch.tensor([-2.0, 1.0, -1.0])
        assert frd.delta(x, in_place=False).tolist() == pytest.approx([-2.0, 3.0, 1.0])

    def test_1d_shape_preserved(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 2.0, 3.0])
        assert frd.delta(x, in_place=False).shape == x.shape

    def test_2d_shape_preserved(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert frd.delta(x, in_place=False).shape == x.shape

    def test_2d_flattened_then_reshaped(self):
        # flatten [1,2,3,4]; first=1; rest: 2-1=1, 3-1=2, 4-1=3 → reshape [[1,1],[2,3]]
        frd = FixedReferenceDelta()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = frd.delta(x, in_place=False)
        assert result[0].tolist() == pytest.approx([1.0, 1.0])
        assert result[1].tolist() == pytest.approx([2.0, 3.0])

    def test_in_place_false_does_not_modify_input(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 3.0, 6.0])
        original = x.clone()
        frd.delta(x, in_place=False)
        assert torch.equal(x, original)

    def test_in_place_true_modifies_input(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 2.0, 3.0])
        frd.delta(x, in_place=True)
        assert x.tolist() == pytest.approx([1.0, 1.0, 2.0])


class TestReconstruct:
    def test_single_element_unchanged(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([7.0])
        assert frd.reconstruct(x, in_place=False).item() == pytest.approx(7.0)

    def test_first_element_unchanged(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([5.0, 1.0, 2.0])
        assert frd.reconstruct(x, in_place=False)[0].item() == pytest.approx(5.0)

    def test_adds_first_element_to_all_rest(self):
        # first=1; rest: 1+1=2, 2+1=3, 3+1=4
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 1.0, 2.0, 3.0])
        assert frd.reconstruct(x, in_place=False).tolist() == pytest.approx(
            [1.0, 2.0, 3.0, 4.0]
        )

    def test_negative_first_element(self):
        # first=-2; rest: 3+(-2)=1, 1+(-2)=-1
        frd = FixedReferenceDelta()
        x = torch.tensor([-2.0, 3.0, 1.0])
        assert frd.reconstruct(x, in_place=False).tolist() == pytest.approx(
            [-2.0, 1.0, -1.0]
        )

    def test_zero_deltas_produce_repeated_first_value(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([3.0, 0.0, 0.0, 0.0])
        assert frd.reconstruct(x, in_place=False).tolist() == pytest.approx(
            [3.0, 3.0, 3.0, 3.0]
        )

    def test_1d_shape_preserved(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 1.0, 2.0])
        assert frd.reconstruct(x, in_place=False).shape == x.shape

    def test_2d_shape_preserved(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
        assert frd.reconstruct(x, in_place=False).shape == x.shape

    def test_2d_flattened_reconstruct_then_reshaped(self):
        # flatten [1,1,2,3]; first=1; rest: 1+1=2, 2+1=3, 3+1=4 → reshape [[1,2],[3,4]]
        frd = FixedReferenceDelta()
        x = torch.tensor([[1.0, 1.0], [2.0, 3.0]])
        result = frd.reconstruct(x, in_place=False)
        assert result[0].tolist() == pytest.approx([1.0, 2.0])
        assert result[1].tolist() == pytest.approx([3.0, 4.0])

    def test_in_place_false_does_not_modify_input(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 1.0, 2.0])
        original = x.clone()
        frd.reconstruct(x, in_place=False)
        assert torch.equal(x, original)

    def test_in_place_true_modifies_input(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([1.0, 1.0, 2.0])
        frd.reconstruct(x, in_place=True)
        assert x.tolist() == pytest.approx([1.0, 2.0, 3.0])


class TestRoundTrip:
    def test_reconstruct_inverts_delta_1d(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([3.0, 7.0, 2.0, 9.0])
        result = frd.reconstruct(frd.delta(x, in_place=False), in_place=False)
        assert result.tolist() == pytest.approx(x.tolist())

    def test_reconstruct_inverts_delta_2d(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([[1.0, 5.0], [3.0, 8.0]])
        result = frd.reconstruct(frd.delta(x, in_place=False), in_place=False)
        assert result[0].tolist() == pytest.approx(x[0].tolist())
        assert result[1].tolist() == pytest.approx(x[1].tolist())

    def test_delta_inverts_reconstruct(self):
        frd = FixedReferenceDelta()
        x = torch.tensor([2.0, 1.0, 4.0, 7.0])
        result = frd.delta(frd.reconstruct(x, in_place=False), in_place=False)
        assert result.tolist() == pytest.approx(x.tolist())
