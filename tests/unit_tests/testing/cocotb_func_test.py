import pytest

from elasticai.creator.testing.cocotb_func import check_results


@pytest.mark.parametrize(
    "result, expected, tol, passed",
    [
        ([1, 2, 3], [1, 2, 3], 0, True),
        ([1, 2, 3], [0, 3, 4], 1, True),
        ([1, 2, 5], [0, 2, 3], 1, False),
        ([-3, -10, -1], [-2, -9, -1], 1, True),
        ([-5, -1], [-2, -1], 1, False),
        ([5], [3], 2, True),
        ([6], [3], 2, False),
        ([10, -10], [15, -15], 5, True),
        ([10, -10], [16, -16], 5, False),
        ([1, 2, 100], [1, 2, 3], 1, False),
        ([], [], 1, True),
        ([1], [1], -1, False),
        ([0, 1, -1, 1000, -1000], [0, 1, -1, 1000, -1000], 0, True),
    ],
)
def test_check_results(
    result: list[int], expected: list[int], tol: int, passed: bool
) -> None:
    assert check_results(result, expected, tol=tol) is passed


@pytest.mark.parametrize("result, expected", [([1, 2, 3], [1, 2]), ([1], [1, 2, 3])])
def test_check_results_length_mismatch(result: list[int], expected: list[int]) -> None:
    with pytest.raises(AssertionError, match="Length mismatch"):
        check_results(result, expected)
