import pytest

from .code_abstractions import to_vhdl_binary_string


def test_to_vhdl_binary_string_raises_error_if_value_not_representable() -> None:
    with pytest.raises(ValueError):
        _ = to_vhdl_binary_string(-8, 3)


@pytest.mark.parametrize(
    ("number", "number_of_bits", "expected"),
    [
        (0, 1, '"0"'),
        (1, 1, '"1"'),
        (6, 4, '"0110"'),
        (-6, 4, '"1010"'),
        (7, 5, '"00111"'),
        (-7, 5, '"11001"'),
    ],
)
def test_to_vhdl_binary_string(number: int, number_of_bits: int, expected: str) -> None:
    assert to_vhdl_binary_string(number, number_of_bits) == expected
