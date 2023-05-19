from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import repeat
from string import Template as StringTemplate
from typing import Protocol, cast

from elasticai.creator.resource_utils import read_text


def module_to_package(module: str) -> str:
    return ".".join(module.split(".")[:-1])


class Template(Protocol):
    parameters: dict[str, str | list[str]]
    content: list[str]


@dataclass
class InProjectTemplate(Template):
    package: str
    file_name: str
    parameters: dict[str, str | list[str]]

    def __post_init__(self) -> None:
        self.content = list(read_text(self.package, self.file_name))


class TemplateExpander:
    def __init__(self, template: Template) -> None:
        super().__init__()
        self._template = template

    def lines(self) -> list[str]:
        self._assert_all_variables_to_fill_exists()

        single_line_params, multi_line_params = _split_single_and_multiline_parameters(
            self._template.parameters
        )
        lines = _expand_template(self._template.content, **single_line_params)
        lines = _expand_multiline_template(lines, **multi_line_params)
        return list(lines)

    def unfilled_variables(self) -> set[str]:
        template_variables = _extract_template_variables(self._template.content)
        variables_to_fill = set(self._template.parameters.keys())
        return template_variables - variables_to_fill

    def _assert_all_variables_to_fill_exists(self) -> None:
        template_variables = _extract_template_variables(self._template.content)
        variables_to_fill = set(self._template.parameters.keys())
        not_existing_variables = variables_to_fill - template_variables
        if len(not_existing_variables) > 0:
            raise KeyError(
                "The template has no variables named"
                f" {', '.join(not_existing_variables)} to fill."
            )


def _split_single_and_multiline_parameters(
    parameters: Mapping[str, str | Iterable[str]],
) -> tuple[dict[str, str], dict[str, Iterator[str]]]:
    single_line_parameters = dict(
        cast(
            Iterator[tuple[str, str]],
            filter(lambda i: isinstance(i[1], str), parameters.items()),
        )
    )
    multiline_parameters = dict(
        cast(
            Iterator[tuple[str, Iterator[str]]],
            filter(lambda i: not isinstance(i[1], str), parameters.items()),
        )
    )
    return single_line_parameters, multiline_parameters


def _expand_multiline_template(
    template: str | Iterable[str], **kwargs: Iterable[str]
) -> Iterator[str]:
    """Expand a template field to multiple lines, while keeping indentation.
    Example:
        >>> template = "\\t$my_key"
        >>> values = ["hello,", "world", "!"]
        >>> "\\n".join(_expand_multiline_template(template, my_key=values))
        '\\thello,\\n\\tworld\\n\\t!'
    """
    lines = _unify_template_datatype(template)
    for line in lines:
        contains_no_key = True
        for key in kwargs:
            if f"${key}" in line:
                contains_no_key = False
                for placeholder_line, value in zip(repeat(line), kwargs[key]):
                    t = StringTemplate(placeholder_line)
                    yield t.safe_substitute({key: value})
                break
        if contains_no_key:
            yield line


def _unify_template_datatype(template: str | Iterable[str]) -> Iterator[str]:
    if isinstance(template, str):
        lines = template.splitlines()
    else:
        lines = list(template)
    yield from lines


def _expand_template(template: str | Iterable[str], **kwargs: str) -> Iterator[str]:
    for line in _unify_template_datatype(template):
        yield StringTemplate(line).safe_substitute(kwargs)


def _extract_template_variables(template: list[str]) -> set[str]:
    return set(StringTemplate("\n".join(template)).get_identifiers())
