from itertools import chain, repeat
from string import Template
from typing import Callable, Iterable, Iterator, Union

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.code import Code, TemplateCodeFile


class VHDLFile(TemplateCodeFile):
    """
    `VHDLFile` helps you to fill templates with values.
    It loads a vhdl template file from the package `elasticai.creator.vhdl.template` and fills template parameters
    with the parameters passed to the constructor.
    Each of these parameters is expected to be either a `str` or a `Iterable[str]`.
    Depending on the type, the parameter will be subject to multi or single line expansion.
    Multi line expansion in the context means the template `"$myValue"` with the parameter `{"myValue": ["1", "2"]}`
    will result in the string `"1\n2"`.
    The **code** method returns the lines of code resulting from the parameters filled into the template.
    """

    _template_package = "elasticai.creator.vhdl.templates"

    def __init__(self, name: str, **parameters: Union[str, Iterable[str]]) -> None:
        self._name = name
        (
            self._parameters,
            self._multiline_parameters,
        ) = self._split_single_and_multiline_parameters(parameters)

    @staticmethod
    def _split_single_and_multiline_parameters(parameters: dict):
        single_line_parameters = dict(
            filter(lambda i: isinstance(i[1], str), parameters.items())
        )
        multiline_parameters = dict(
            filter(lambda i: not isinstance(i[1], str), parameters.items())
        )
        return single_line_parameters, multiline_parameters

    @property
    def single_line_parameters(self) -> dict[str, str]:
        return dict(**self._parameters)

    @property
    def parameters(self):
        return dict(chain(self._parameters.items(), self._multiline_parameters.items()))

    @property
    def multi_line_parameters(self):
        return dict(**self._multiline_parameters)

    @property
    def name(self) -> str:
        return f"{self._name}.vhd"

    def code(self) -> Code:
        template = read_text(self._template_package, f"{self._name}.tpl.vhd")
        template = expand_template(template, **self._parameters)
        template = expand_multiline_template(template, **self._multiline_parameters)
        return template


def expand_multiline_template(
    template: Union[str, Iterable[str]], **kwargs: Iterable[str]
) -> Iterator[str]:
    """Expand a template field to multiple lines, while keeping indentation.
    Example:
        >>> template = "\\t$my_key"
        >>> values = ["hello,", "world", "!"]
        >>> "\\n".join(expand_multiline_template(template, my_key=values))
        '\\thello,\\n\\tworld\\n\\t!'
    """
    lines = _unify_template_datatype(template)
    for line in lines:
        contains_no_key = True
        for key in kwargs:
            if f"${key}" in line:
                contains_no_key = False
                for placeholder_line, value in zip(repeat(line), kwargs[key]):
                    t = Template(placeholder_line)
                    yield t.safe_substitute({key: value})
                break
        if contains_no_key:
            yield line


def _unify_template_datatype(template: Union[str, Iterable[str]]) -> Iterable[str]:
    if hasattr(template, "splitlines") and callable(template.splitlines):
        lines = template.splitlines()
    else:
        lines = template
    return lines


def expand_template(template: str | Iterable[str], **kwargs: str) -> Iterable[str]:
    if isinstance(template, str):
        yield Template(template).safe_substitute(kwargs)
    else:
        for line in template:
            yield Template(line).safe_substitute(kwargs)
