import pytest
import torch

from elasticai.creator.experimental.delta_compression.src.builder import (
    DeltaCompBuilder,
)
from elasticai.creator.experimental.delta_compression.src.delta_compression import (
    DeltaCompression,
)


def _make_dc() -> DeltaCompression:
    return (
        DeltaCompBuilder()
        .consecutive_delta()
        .saturated_compression(delta_width=4, offset=0)
        .build()
    )


# consecutive_delta + saturated_compression(width=4, offset=0):
#   compress: delta-encode → clamp abs to [0, 7], preserve sign
#   inflate:  cumsum (sat.inflate is identity)


class TestCompress:
    def test_output_shape_preserved(self):
        dc = _make_dc()
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert dc.compress(x, in_place=False).shape == x.shape

    def test_first_element_unchanged_when_within_range(self):
        dc = _make_dc()
        x = torch.tensor([5.0, 6.0, 7.0])
        result = dc.compress(x, in_place=False)
        assert result[0].item() == pytest.approx(5.0)

    def test_consecutive_equal_values_produce_zero_deltas(self):
        dc = _make_dc()
        x = torch.tensor([3.0, 3.0, 3.0, 3.0])
        result = dc.compress(x, in_place=False)
        assert result[1:].tolist() == pytest.approx([0.0, 0.0, 0.0])

    def test_increasing_values_within_range(self):
        # deltas [1,1,1,1] all ≤ 7 → unchanged
        dc = _make_dc()
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = dc.compress(x, in_place=False)
        assert result.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_large_delta_saturates_to_max(self):
        # delta = 1000 → clamped to 7
        dc = _make_dc()
        x = torch.tensor([0.0, 1000.0])
        result = dc.compress(x, in_place=False)
        assert result[1].item() == pytest.approx(7.0)

    def test_negative_delta_preserved_within_range(self):
        # delta = 3 - 5 = -2, abs(-2)=2 ≤ 7 → unchanged
        dc = _make_dc()
        x = torch.tensor([5.0, 3.0])
        result = dc.compress(x, in_place=False)
        assert result[1].item() == pytest.approx(-2.0)

    def test_large_negative_delta_saturates(self):
        # delta = -1000 → clamped abs to 7, sign preserved → -7
        dc = _make_dc()
        x = torch.tensor([0.0, -1000.0])
        result = dc.compress(x, in_place=False)
        assert result[1].item() == pytest.approx(-7.0)

    def test_in_place_false_does_not_modify_input(self):
        dc = _make_dc()
        x = torch.tensor([1.0, 2.0, 3.0])
        original = x.clone()
        dc.compress(x, in_place=False)
        assert torch.equal(x, original)


class TestInflate:
    def test_output_shape_preserved(self):
        dc = _make_dc()
        x = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
        assert dc.inflate(x, in_place=False).shape == x.shape

    def test_cumsum_reconstructs_values(self):
        # inflate([1, 1, 1, 1]) = cumsum = [1, 2, 3, 4]
        dc = _make_dc()
        x = torch.tensor([1.0, 1.0, 1.0, 1.0])
        result = dc.inflate(x, in_place=False)
        assert result.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0])

    def test_all_zero_deltas_repeat_first_value(self):
        dc = _make_dc()
        x = torch.tensor([5.0, 0.0, 0.0, 0.0])
        result = dc.inflate(x, in_place=False)
        assert result.tolist() == pytest.approx([5.0, 5.0, 5.0, 5.0])

    def test_in_place_false_does_not_modify_input(self):
        dc = _make_dc()
        x = torch.tensor([1.0, 1.0, 1.0])
        original = x.clone()
        dc.inflate(x, in_place=False)
        assert torch.equal(x, original)

    def test_round_trip_within_compression_range(self):
        # all consecutive deltas ≤ 7 → inflate(compress(x)) == x
        dc = _make_dc()
        x = torch.tensor([1.0, 2.0, 4.0, 6.0])
        result = dc.inflate(dc.compress(x, in_place=False), in_place=False)
        assert result.tolist() == pytest.approx(x.tolist())

    def test_round_trip_outside_range_differs_from_original(self):
        # delta = 1000 → saturated → inflate gives different value
        dc = _make_dc()
        x = torch.tensor([0.0, 1000.0])
        result = dc.inflate(dc.compress(x, in_place=False), in_place=False)
        assert not torch.allclose(result, x)
