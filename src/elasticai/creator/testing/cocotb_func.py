def check_results(result: list[int], expected: list[int], tol: int = 1) -> bool:
    assert len(result) == len(expected), "Length mismatch"
    return all(abs(r - e) <= tol for r, e in zip(result, expected))
