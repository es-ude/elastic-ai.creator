def all_within_tolerance(result: list[int], expected: list[int], tol: int = 1) -> bool:
    if len(result) != len(expected):
        raise ValueError("Length mismatch")
    return all(abs(r - e) <= tol for r, e in zip(result, expected))


def is_mae_within_tolerance(
    result: list[int], expected: list[int], tol: float = 0.5
) -> bool:
    if len(result) != len(expected):
        raise ValueError("Length mismatch")
    if not result:
        raise ValueError("Empty input")
    mae = sum(abs(r - e) for r, e in zip(result, expected)) / len(result)
    return mae <= tol


def error_count_outside_tolerance(
    result: list[int], expected: list[int], tol: int = 0
) -> int:
    if len(result) != len(expected):
        raise ValueError("Length mismatch")
    return sum(abs(r - e) > tol for r, e in zip(result, expected))
