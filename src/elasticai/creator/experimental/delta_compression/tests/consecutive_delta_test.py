import pytest
import torch

from elasticai.creator.experimental.delta_compression.src.delta.consecutive_delta import (
    ConsecutiveDelta,
)

# ConsecutiveDelta:
#   delta:       first element unchanged; rest become input[i] - input[i-1]
#   reconstruct: cumsum restores originals (first unchanged; rest are running sums)
#   Both flatten internally and reshape to original shape.


class TestDelta:
    def test_single_element_unchanged(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([7.0])
        assert cd.delta(x, in_place=False).item() == pytest.approx(7.0)

    def test_first_element_always_unchanged(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([5.0, 1.0, 2.0])
        assert cd.delta(x, in_place=False)[0].item() == pytest.approx(5.0)

    def test_increasing_sequence_produces_positive_deltas(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert cd.delta(x, in_place=False).tolist() == pytest.approx(
            [1.0, 1.0, 1.0, 1.0]
        )

    def test_decreasing_sequence_produces_negative_deltas(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert cd.delta(x, in_place=False).tolist() == pytest.approx(
            [4.0, -1.0, -1.0, -1.0]
        )

    def test_equal_values_produce_zero_deltas(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([3.0, 3.0, 3.0, 3.0])
        assert cd.delta(x, in_place=False).tolist() == pytest.approx(
            [3.0, 0.0, 0.0, 0.0]
        )

    def test_mixed_positive_and_negative_values(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([-1.0, 2.0, -1.0])
        # deltas: -1, 2-(-1)=3, -1-2=-3
        assert cd.delta(x, in_place=False).tolist() == pytest.approx([-1.0, 3.0, -3.0])

    def test_non_unit_steps(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([0.0, 0.5, 1.5, 3.0])
        assert cd.delta(x, in_place=False).tolist() == pytest.approx(
            [0.0, 0.5, 1.0, 1.5]
        )

    def test_1d_shape_preserved(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 2.0, 3.0])
        assert cd.delta(x, in_place=False).shape == x.shape

    def test_2d_shape_preserved(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([[1.0, 3.0], [6.0, 10.0]])
        assert cd.delta(x, in_place=False).shape == x.shape

    def test_2d_flattened_then_reshaped(self):
        # flatten [1,3,6,10] → deltas [1,2,3,4] → reshape [[1,2],[3,4]]
        cd = ConsecutiveDelta()
        x = torch.tensor([[1.0, 3.0], [6.0, 10.0]])
        result = cd.delta(x, in_place=False)
        assert result[0].tolist() == pytest.approx([1.0, 2.0])
        assert result[1].tolist() == pytest.approx([3.0, 4.0])

    def test_in_place_false_does_not_modify_input(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 2.0, 3.0])
        original = x.clone()
        cd.delta(x, in_place=False)
        assert torch.equal(x, original)

    def test_in_place_true_modifies_input(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 2.0, 3.0])
        cd.delta(x, in_place=True)
        assert x.tolist() == pytest.approx([1.0, 1.0, 1.0])


class TestReconstruct:
    def test_single_element_unchanged(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([7.0])
        assert cd.reconstruct(x, in_place=False).item() == pytest.approx(7.0)

    def test_first_element_unchanged(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([5.0, 1.0, 1.0])
        assert cd.reconstruct(x, in_place=False)[0].item() == pytest.approx(5.0)

    def test_cumsum_of_unit_deltas(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 1.0, 1.0, 1.0])
        assert cd.reconstruct(x, in_place=False).tolist() == pytest.approx(
            [1.0, 2.0, 3.0, 4.0]
        )

    def test_cumsum_of_negative_deltas(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([4.0, -1.0, -1.0, -1.0])
        assert cd.reconstruct(x, in_place=False).tolist() == pytest.approx(
            [4.0, 3.0, 2.0, 1.0]
        )

    def test_zero_deltas_repeat_first_value(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([3.0, 0.0, 0.0, 0.0])
        assert cd.reconstruct(x, in_place=False).tolist() == pytest.approx(
            [3.0, 3.0, 3.0, 3.0]
        )

    def test_1d_shape_preserved(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 1.0, 1.0])
        assert cd.reconstruct(x, in_place=False).shape == x.shape

    def test_2d_shape_preserved(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert cd.reconstruct(x, in_place=False).shape == x.shape

    def test_2d_flattened_cumsum_then_reshaped(self):
        # flatten [1,2,3,4] → cumsum [1,3,6,10] → reshape [[1,3],[6,10]]
        cd = ConsecutiveDelta()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = cd.reconstruct(x, in_place=False)
        assert result[0].tolist() == pytest.approx([1.0, 3.0])
        assert result[1].tolist() == pytest.approx([6.0, 10.0])

    def test_in_place_false_does_not_modify_input(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 1.0, 1.0])
        original = x.clone()
        cd.reconstruct(x, in_place=False)
        assert torch.equal(x, original)

    def test_in_place_true_modifies_input(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 1.0, 1.0])
        cd.reconstruct(x, in_place=True)
        assert x.tolist() == pytest.approx([1.0, 2.0, 3.0])


class TestRoundTrip:
    def test_reconstruct_inverts_delta_1d(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([3.0, 7.0, 2.0, 9.0])
        result = cd.reconstruct(cd.delta(x, in_place=False), in_place=False)
        assert result.tolist() == pytest.approx(x.tolist())

    def test_reconstruct_inverts_delta_2d(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([[1.0, 5.0], [3.0, 8.0]])
        result = cd.reconstruct(cd.delta(x, in_place=False), in_place=False)
        assert result[0].tolist() == pytest.approx(x[0].tolist())
        assert result[1].tolist() == pytest.approx(x[1].tolist())

    def test_delta_inverts_reconstruct(self):
        cd = ConsecutiveDelta()
        x = torch.tensor([1.0, 2.0, 4.0, 7.0])
        result = cd.delta(cd.reconstruct(x, in_place=False), in_place=False)
        assert result.tolist() == pytest.approx(x.tolist())
