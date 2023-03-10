from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, repeat
from string import Template as StringTemplate
from typing import Iterable, Iterator, Mapping, cast

from elasticai.creator.resource_utils import read_text


class AbstractBaseTemplate(ABC):
    """
    `Template` helps you to fill template_resources with values.
    It loads a vhdl template file from the package `elasticai.creator.vhdl.template` and fills template parameters
    with the parameters passed to the constructor.
    Each of these parameters is expected to be either a `str` or a `Iterable[str]`.
    Depending on the type, the parameter will be subject to multi or single line expansion.
    Multi line expansion in the context means the template `"$myValue"` with the parameter `{"myValue": ["1", "2"]}`
    will result in the string `"1\n2"`.
    The **code** method returns the lines of code resulting from the parameters filled into the template.
    """

    def __init__(self, **parameters: str | list[str] | tuple[str]) -> None:
        self._parameters: dict[str, str] = dict()
        self._multiline_parameters: dict[str, Iterable[str]] = dict()
        self.update_parameters(**parameters)
        self._saved_raw_template: list[str] = []

    @abstractmethod
    def _read_raw_template(self) -> Iterator[str]:
        ...

    @property
    def _raw_template(self) -> list[str]:
        if len(self._saved_raw_template) == 0:
            self._saved_raw_template = list(self._read_raw_template())
        return self._saved_raw_template

    def update_parameters(self, **parameters: str | Iterable[str]) -> None:
        (
            single_line_parameters,
            multiline_parameters,
        ) = _split_single_and_multiline_parameters(parameters)
        self._parameters.update(**single_line_parameters)
        self._multiline_parameters.update(**multiline_parameters)

    @property
    def single_line_parameters(self) -> dict[str, str]:
        return dict(**self._parameters)

    @property
    def parameters(self):
        single_items = self._parameters.items()
        multi_items = self._multiline_parameters.items()
        return dict(
            cast(
                Iterator[tuple[str, str] | tuple[str, Iterable[str]]],
                chain(single_items, multi_items),
            )
        )

    @property
    def multi_line_parameters(self):
        return dict(**self._multiline_parameters)

    def lines(self) -> list[str]:
        lines = _expand_template(self._raw_template, **self._parameters)
        lines = _expand_multiline_template(lines, **self._multiline_parameters)
        return list(lines)


def module_to_package(module: str) -> str:
    return ".".join(module.split(".")[:-1])


@dataclass
class TemplateConfig:
    """
    Used in design definition, by the hw designer.
    HW designer just provides template configs and port definitions as a design.

    Contributor of a new translatable module provides design and ml module as well as how to map parameters

    Creator takes these and uses the template expander to generate the correct file
    """

    package: str
    file_name: str
    parameters: dict[str, str | list[str]]


class TemplateExpander:
    """
    Used during translation by the creator tool. HW designer does not need to touch this or inherit from it.
    """

    def _read_raw_template(self) -> Iterator[str]:
        return read_text(
            self.config.package,
            self.config.file_name,
        )

    def __init__(self, config: TemplateConfig):
        super().__init__()
        self.config = config

    def lines(self) -> list[str]:
        single_line_params, multi_line_params = _split_single_and_multiline_parameters(
            self.config.parameters
        )
        template = self._read_raw_template()
        _lines = _expand_template(template, **single_line_params)
        _lines = _expand_multiline_template(_lines, **multi_line_params)
        return list(_lines)


def _split_single_and_multiline_parameters(
    parameters: Mapping[str, str | Iterable[str]],
) -> tuple[dict[str, str], dict[str, Iterable[str]]]:
    single_line_parameters: dict[str, str] = dict(
        cast(
            Iterator[tuple[str, str]],
            filter(lambda i: isinstance(i[1], str), parameters.items()),
        )
    )
    multiline_parameters = dict(
        cast(
            Iterator[tuple[str, Iterable[str]]],
            filter(lambda i: not isinstance(i[1], str), parameters.items()),
        )
    )
    return single_line_parameters, multiline_parameters


def _expand_multiline_template(
    template: str | list[str] | Iterator[str], **kwargs: Iterable[str]
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


def _unify_template_datatype(
    template: str | list[str] | Iterator[str],
) -> Iterator[str]:
    if hasattr(template, "splitlines") and callable(template.splitlines):
        lines = template.splitlines()
    else:
        lines = template
    yield from lines


def _expand_template(
    template: str | list[str] | Iterator[str], **kwargs: str
) -> Iterator[str]:
    for line in _unify_template_datatype(template):
        yield StringTemplate(line).safe_substitute(kwargs)
