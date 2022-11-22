import doctest
from collections.abc import Iterator
from unittest import TestCase

import elasticai.creator.vhdl.templates.utils
from elasticai.creator.vhdl.templates.utils import (
    expand_multiline_template,
    expand_template,
)


def newline_join(lines: Iterator[str]) -> str:
    return "\n".join(lines)


class ExpandTemplatesTestCase(TestCase):
    @staticmethod
    def get_result_string(template, **kwargs) -> str:
        return newline_join(expand_multiline_template(template, **kwargs))

    def test_expand_multiline_with_one_item(self):
        # noinspection PyShadowingNames
        def check_single_item(value: str):
            template = "$my_var"
            expected = value
            values = [value]
            actual = self.get_result_string(template, my_var=values)
            self.assertEqual(expected, actual)

        for value in ["value", "some other value"]:
            with self.subTest():
                check_single_item(value)

    def single_value_with_different_key(self):
        template = "$my_key"
        expected = "value"
        actual = self.get_result_string(template, my_key=[expected])
        self.assertEqual(expected, actual)

    def test_expand_two_keys(self):
        template = "$first\n$second"
        expected = "ab\ncd"
        actual = "\n".join(
            list(expand_multiline_template(template, first=("ab",), second=("cd",)))
        )
        self.assertEqual(expected, actual)

    def test_two_values(self):
        template = "$key"
        expected = "a\nb"
        actual = self.get_result_string(template, key=["a", "b"])
        self.assertEqual(expected, actual)

    def test_no_values(self):
        template = "$key"
        expected = ""
        actual = self.get_result_string(template, key=[])
        self.assertEqual(expected, actual)

    def test_key_not_in_template(self):
        template = "something"
        expected = "something"
        actual = self.get_result_string(template, key=["a"])
        self.assertEqual(expected, actual)

    def test_keep_indentation(self):
        template = "\t  $key"
        expected = "\t  a\n\t  b\n\t  c"
        actual = "\n".join(
            list(expand_multiline_template(template, key=["a", "b", "c"]))
        )
        self.assertEqual(expected, actual)

    def test_no_error_if_not_all_keys_are_filled(self):
        template = "$key"
        # noinspection PyBroadException
        try:
            actual = self.get_result_string(template, other_key=[])
            self.assertEqual(template, actual)
        except Exception:
            self.fail()

    def test_two_keys_on_one_line(self):
        template = "\t$first $second"
        values = [0, 1]
        expected = "\t0 $second\n\t1 $second"
        actual = self.get_result_string(template, first=values, second=values)
        self.assertEqual(expected, actual)

    def test_expand_two_keys_by_running_twice(self):
        template = "\t$first $second"
        values = [0, 1]
        expected = "\t0 0\n\t0 1\n\t1 0\n\t1 1"
        actual = self.get_result_string(template, first=values)
        actual = self.get_result_string(actual, first=values, second=values)
        self.assertEqual(expected, actual)


class ExpandTemplatesTest(TestCase):
    def test_expand_empty_string_template(self) -> None:
        template = ""
        actual = newline_join(expand_template(template, a=1, b="hello"))
        expected = ""
        self.assertEqual(expected, actual)

    def test_expand_single_string_template_with_single_key(self) -> None:
        template = "$some_key"
        actual = newline_join(expand_template(template, some_key=42))
        expected = "42"
        self.assertEqual(expected, actual)

    def test_raise_key_error_for_missing_key(self) -> None:
        template = "$some_key"
        with self.assertRaises(KeyError):
            _ = newline_join(expand_template(template))

    def test_expand_multiple_strings_template(self) -> None:
        template = ["$val1", "$val2", "$val3"]
        actual = newline_join(
            expand_template(template, val1="hello", val2="world", val3=42)
        )
        expected = "hello\nworld\n42"
        self.assertEqual(expected, actual)

    def test_expand_multiple_strings_template_with_multiple_keys_per_line(self) -> None:
        template = ["$val1 $val1 $val1", "$val1 $val2", ""]
        actual = newline_join(
            expand_template(template, val1="hello", val2="world", val3=42)
        )
        expected = "hello hello hello\nhello world\n"
        self.assertEqual(expected, actual)


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(elasticai.creator.vhdl.templates.utils))
    return tests
