import pytest

from elasticai.creator.vhdl.code_generation.code_abstractions import (
    to_vhdl_binary_string,
)


@pytest.mark.parametrize(
    ("number", "number_of_bits"),
    [
        (2, 1),
        (-5, 3),
        (4, 3),
    ],
)
def test_to_vhdl_binary_string_raises_error_if_value_not_representable(
    number: int, number_of_bits: int
) -> None:
    with pytest.raises(ValueError):
        _ = to_vhdl_binary_string(number, number_of_bits)


@pytest.mark.parametrize(
    ("number", "number_of_bits", "expected"),
    [
        (1, 2, '"01"'),
        (0, 2, '"00"'),
        (6, 4, '"0110"'),
        (-6, 4, '"1010"'),
        (7, 5, '"00111"'),
        (-7, 5, '"11001"'),
        (-4, 3, '"100"'),
        (3, 3, '"011"'),
    ],
)
def test_to_vhdl_binary_string(number: int, number_of_bits: int, expected: str) -> None:
    assert to_vhdl_binary_string(number, number_of_bits) == expected
