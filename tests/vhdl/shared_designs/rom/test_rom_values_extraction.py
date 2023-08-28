import pytest

from .utils import extract_rom_values


@pytest.mark.parametrize(
    ("values_to_extract", "values"),
    [
        ('"0"', ("0",)),
        ('"0", "1"', ("0", "1")),
        ('"10101111", "10111011"', ("10101111", "10111011")),
        ('"10101111","10111011"', ("10101111", "10111011")),
    ],
)
def test_extracted_rom_values(values_to_extract: str, values: tuple[str]) -> None:
    test_line = f"signal ROM : some_name_array_t:=({values_to_extract})"
    assert extract_rom_values(test_line) == values


def test_can_handle_iteration_over_multiple_lines():
    test_lines = [
        "some text",
        'some text that is comparable to array ("0", "1")',
        'some_other_name_array_t:=("10101111", "10111011")',
    ]
    assert extract_rom_values(test_lines) == ("10101111", "10111011")
