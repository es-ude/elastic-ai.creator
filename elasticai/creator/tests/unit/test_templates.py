import doctest
from abc import ABC
from collections.abc import Iterator
from typing import Iterable
from unittest import TestCase

from elasticai.creator.templates import AbstractBaseTemplate
from elasticai.creator.vhdl.language import vhdl_template


class SimpleTemplate(AbstractBaseTemplate, ABC):
    def __init__(
        self, raw_template: list[str], **parameters: str | tuple[str] | list[str]
    ):
        super().__init__(**parameters)
        self.__raw_template = raw_template

    def _read_raw_template(self) -> Iterator[str]:
        yield from self.__raw_template


def newline_join(lines: Iterable[str]) -> str:
    return "\n".join(lines)


def expand_template(
    template: list[str], **parameters: str | tuple[str] | list[str]
) -> list[str]:
    return SimpleTemplate(template, **parameters).lines()


class ExpandTemplatesTestCase(TestCase):
    @staticmethod
    def get_result_string(
        template: list[str], **kwargs: str | tuple[str] | list[str]
    ) -> str:
        return newline_join(expand_template(template, **kwargs))

    def test_handles_gracefully_list_of_lines(self):
        template = ["my", "$value"]
        expected = "my\nvalue"
        actual = self.get_result_string(template, value=["value"])
        self.assertEqual(expected, actual)

    def test_expand_multiline_with_one_item(self):
        # noinspection PyShadowingNames
        def check_single_item(value: str):
            template = ["$my_var"]
            expected = value
            values = [value]
            actual = self.get_result_string(template, my_var=values)
            self.assertEqual(expected, actual)

        for value in ["value", "some other value"]:
            with self.subTest():
                check_single_item(value)

    def single_value_with_different_key(self):
        template = "$my_key".splitlines()
        expected = "value"
        actual = self.get_result_string(template, my_key=[expected])
        self.assertEqual(expected, actual)

    def test_expand_two_keys(self):
        template = "$first\n$second".splitlines()
        expected = "ab\ncd"
        actual = "\n".join(
            SimpleTemplate(template, first=("ab",), second=("cd",)).lines()
        )
        self.assertEqual(expected, actual)

    def test_two_values(self):
        template = "$key".splitlines()
        expected = "a\nb"
        actual = self.get_result_string(template, key=["a", "b"])
        self.assertEqual(expected, actual)

    def test_no_values(self):
        template = "$key".splitlines()
        expected = ""
        actual = self.get_result_string(template, key=[])
        self.assertEqual(expected, actual)

    def test_key_not_in_template(self):
        template = "something".splitlines()
        expected = "something"
        actual = self.get_result_string(template, key=["a"])
        self.assertEqual(expected, actual)

    def test_keep_indentation(self):
        template = "\t  $key".splitlines()
        expected = "\t  a\n\t  b\n\t  c"
        actual = "\n".join(list(SimpleTemplate(template, key=["a", "b", "c"]).lines()))
        self.assertEqual(expected, actual)

    def test_no_error_if_not_all_keys_are_filled(self):
        template = "$key".splitlines()
        # noinspection PyBroadException
        try:
            actual = expand_template(template, other_key=[])
            self.assertEqual(template, actual)
        except Exception:
            self.fail()

    def test_two_keys_on_one_line(self):
        template = "\t$first $second".splitlines()
        values = ["0", "1"]
        expected = "\t0 $second\n\t1 $second"
        actual = self.get_result_string(template, first=values, second=values)
        self.assertEqual(expected, actual)

    def test_expand_two_keys_by_running_twice(self):
        template = "\t$first $second".splitlines()
        values = ["0", "1"]
        expected = "\t0 0\n\t0 1\n\t1 0\n\t1 1"
        actual = expand_template(template, first=values)
        actual = expand_template(actual, second=values)
        self.assertEqual(expected, newline_join(actual))

    def test_expand_empty_string_template(self) -> None:
        template = "".splitlines()
        actual = newline_join(expand_template(template, a="1", b="hello"))
        expected = ""
        self.assertEqual(expected, actual)

    def test_expand_single_string_template_with_single_key(self) -> None:
        template = "$some_key".splitlines()
        actual = newline_join(expand_template(template, some_key="42"))
        expected = "42"
        self.assertEqual(expected, actual)

    def test_expand_multiple_strings_template(self) -> None:
        template = ["$val1", "$val2", "$val3"]
        actual = newline_join(
            expand_template(template, val1="hello", val2="world", val3="42")
        )
        expected = "hello\nworld\n42"
        self.assertEqual(expected, actual)

    def test_expand_multiple_strings_template_with_multiple_keys_per_line(self) -> None:
        template = ["$val1 $val1 $val1", "$val1 $val2", ""]
        actual = newline_join(
            expand_template(template, val1="hello", val2="world", val3=str(42))
        )
        expected = "hello hello hello\nhello world\n"
        self.assertEqual(expected, actual)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(vhdl_template))
    return tests
