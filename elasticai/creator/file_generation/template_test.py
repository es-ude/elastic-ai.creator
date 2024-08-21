from collections.abc import Iterable
from dataclasses import dataclass

import pytest

from .template import TemplateExpander


@dataclass
class Template:
    content: list[str]
    parameters: dict[str, str | list[str]]


def newline_join(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def expand_template(content: list[str], **parameters: str | list[str]) -> list[str]:
    template = Template(content, parameters)
    return TemplateExpander(template).lines()


def get_result_string(template: list[str], **kwargs: str | list[str]) -> str:
    return newline_join(expand_template(template, **kwargs))


def test_handles_gracefully_list_of_lines() -> None:
    template = ["my", "$value"]
    expected = "my\nvalue"
    actual = get_result_string(template, value=["value"])
    assert expected == actual


@pytest.mark.parametrize("value", ["value", "some other value"])
def test_expand_multiline_with_one_item(value: str) -> None:
    def check_single_item(value: str):
        template = ["$my_var"]
        expected = value
        values = [value]
        actual = get_result_string(template, my_var=values)
        assert expected == actual

    check_single_item(value)


def test_single_value_with_different_key() -> None:
    template = "$my_key".splitlines()
    expected = "value"
    actual = get_result_string(template, my_key=[expected])
    assert expected == actual


def test_expand_two_keys() -> None:
    template = "$first\n$second".splitlines()
    expected = "ab\ncd"
    actual = get_result_string(template, first=["ab"], second=["cd"])
    assert expected == actual


def test_two_values() -> None:
    template = "$key".splitlines()
    expected = "a\nb"
    actual = get_result_string(template, key=["a", "b"])
    assert expected == actual


def test_no_values() -> None:
    template = "$key".splitlines()
    expected = ""
    actual = get_result_string(template, key=[])
    assert expected == actual


def test_keep_indentation() -> None:
    template = "\t  $key".splitlines()
    expected = "\t  a\n\t  b\n\t  c"
    actual = get_result_string(template, key=["a", "b", "c"])
    assert expected == actual


def test_no_error_if_not_all_keys_are_filled() -> None:
    template = "$key\n$other_key".splitlines()
    expected = ["$key"]
    actual = expand_template(template, other_key=[])
    assert expected == actual


def test_two_keys_on_one_line() -> None:
    template = "\t$first $second".splitlines()
    values = ["0", "1"]
    expected = "\t0 $second\n\t1 $second"
    actual = get_result_string(template, first=values, second=values)
    assert expected == actual


def test_expand_two_keys_by_running_twice() -> None:
    template = "\t$first $second".splitlines()
    values = ["0", "1"]
    expected = "\t0 0\n\t0 1\n\t1 0\n\t1 1"
    actual = expand_template(template, first=values)
    actual = expand_template(actual, second=values)
    assert expected == newline_join(actual)


def test_expand_single_string_template_with_single_key() -> None:
    template = "$some_key".splitlines()
    expected = "42"
    actual = get_result_string(template, some_key="42")
    assert expected == actual


def test_expand_multiple_strings_template() -> None:
    template = ["$val1", "$val2", "$val3"]
    expected = "hello\nworld\n42"
    actual = get_result_string(template, val1="hello", val2="world", val3="42")
    assert expected == actual


def test_expand_multiple_strings_template_with_multiple_keys_per_line() -> None:
    template = ["$val1 $val1 $val1", "$val1 $val2", ""]
    expected = "hello hello hello\nhello world\n"
    actual = get_result_string(template, val1="hello", val2="world")
    assert expected == actual


def test_raises_exception_when_inserting_non_existent_keys() -> None:
    template = ["$var1", "$var2"]
    with pytest.raises(KeyError):
        expand_template(template, var3="fail")


def test_variables_not_filled_returns_empty_set() -> None:
    template = Template(["$a", "$b", "$c"], parameters=dict(a="1", b="2", c="3"))
    expander = TemplateExpander(template)
    assert set() == expander.unfilled_variables()


def test_variables_not_filled_returns_non_empty_set() -> None:
    template = Template(["$a", "$b", "$c"], parameters=dict(a="1", c="3"))
    expander = TemplateExpander(template)
    assert {"b"} == expander.unfilled_variables()
