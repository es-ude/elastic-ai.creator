import pytest

from elasticai.creator.testing.cocotb_func import (
    all_within_tolerance,
    error_count_outside_tolerance,
    is_mae_within_tolerance,
)


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
    assert all_within_tolerance(result, expected, tol=tol) is passed


@pytest.mark.parametrize("result, expected", [([1, 2, 3], [1, 2]), ([1], [1, 2, 3])])
def test_check_results_length_mismatch(result: list[int], expected: list[int]) -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        all_within_tolerance(result, expected)


@pytest.mark.parametrize(
    "result, expected, tol, passed",
    [
        ([1, 2, 3], [1, 2, 3], 0, True),
        ([1, 2, 3], [0, 3, 4], 1, True),
        ([1, 2, 5], [0, 2, 3], 1, True),
        ([-3, -10, -1], [-2, -9, -1], 1, True),
        ([-5, -1], [-2, -1], 1, False),
        ([5], [3], 2, True),
        ([6], [3], 2, False),
        ([10, -10], [15, -15], 5, True),
        ([10, -10], [16, -16], 5, False),
        ([1, 2, 100], [1, 2, 3], 1, False),
        ([1], [1], -1, False),
        ([0, 1, -1, 1000, -1000], [0, 1, -1, 1000, -1000], 0, True),
    ],
)
def test_check_mae(
    result: list[int], expected: list[int], tol: int, passed: bool
) -> None:
    assert is_mae_within_tolerance(result, expected, tol=tol) is passed


@pytest.mark.parametrize(
    "result, expected",
    [
        ([1, 2, 3], [1, 2]),
        ([1], [1, 2, 3]),
        ([], []),
    ],
)
def test_mae_length_mismatch(result: list[int], expected: list[int]) -> None:
    with pytest.raises(ValueError):
        is_mae_within_tolerance(result, expected)


@pytest.mark.parametrize(
    "result, expected, tol, passed",
    [
        ([1, 2, 3], [1, 2, 3], 0, 0),
        ([1, 2, 3], [0, 3, 4], 1, 0),
        ([1, 2, 5], [0, 2, 3], 1, 1),
        ([-3, -10, -1], [-2, -9, -1], 1, 0),
        ([-5, -1], [-2, -1], 1, 1),
        ([5], [3], 2, 0),
        ([6], [3], 2, 1),
        ([10, -10], [15, -15], 5, 0),
        ([10, -10], [16, -16], 5, 2),
        ([1, 2, 100], [1, 2, 3], 1, 1),
        ([1], [1], -1, 1),
        ([0, 1, -1, 1000, -1000], [0, 1, -1, 1000, -1000], 0, 0),
    ],
)
def test_error_count(
    result: list[int], expected: list[int], tol: int, passed: int
) -> None:
    assert error_count_outside_tolerance(result, expected, tol=tol) == passed


@pytest.mark.parametrize("result, expected", [([1, 2, 3], [1, 2]), ([1], [1, 2, 3])])
def test_error_count_mismatch(result: list[int], expected: list[int]) -> None:
    with pytest.raises(ValueError):
        error_count_outside_tolerance(result, expected)
