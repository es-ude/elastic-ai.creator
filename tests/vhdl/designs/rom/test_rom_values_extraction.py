import re

from .utils import extract_rom_values


def test_capturing_sequence_repetitions_with_regex():
    repeated_sequence = "amama"
    match = re.split(r"m", repeated_sequence)
    assert match == ["a", "a", "a"]


def test_extracted_rom_values_are_00_01():
    test_line = 'signal ROM : some_name_array_t:=(x"00", x"01")'
    assert extract_rom_values(test_line) == ("00", "01")


def test_can_extract_rom_values_af_fe():
    test_line = 'signal ROM : some_name_array_t:=(x"af", x"fe")'
    assert extract_rom_values(test_line) == ("af", "fe")


def test_can_extract_af_fe_without_space_after_comma():
    test_line = 'signal ROM : some_name_array_t:=(x"af",x"fe")'
    assert extract_rom_values(test_line) == ("af", "fe")


def test_can_extract_af_fe_01():
    test_line = 'signal ROM : some_name_array_t:=(x"af", x"fe", x"01")'
    assert extract_rom_values(test_line) == ("af", "fe", "01")


def test_can_handle_iteration_over_multiple_lines():
    test_lines = [
        "some text",
        'some text that is comparable to array (x"00", x"01")',
        'some_other_name_array_t:=(x"af", x"be")',
    ]
    assert extract_rom_values(test_lines) == ("af", "be")


def test_can_extract_single_value():
    test_line = 'signal ROM : some_array_t:=(x"00");'
    assert extract_rom_values(test_line) == ("00",)


def test_extract_longer_value():
    test_line = 'signal ROM : some_array_t:=(x"0000");'
    assert extract_rom_values(test_line) == ("0000",)
