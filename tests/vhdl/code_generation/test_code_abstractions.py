import pytest

from elasticai.creator.vhdl.code_generation.code_abstractions import (
    to_vhdl_binary_string,
)


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (0, '"0"'),
        (1, '"1"'),
        (6, '"110"'),
        (-6, '"-110"'),
    ],
)
def test_to_vhdl_binary_string(number: int, expected: str) -> None:
    assert to_vhdl_binary_string(number) == expected
