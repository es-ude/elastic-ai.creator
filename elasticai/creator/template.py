from typing import TypeAlias, Protocol, runtime_checkable, AnyStr
from collections.abc import Sequence
from string import Template as _pyTemplate
from re import Match, Pattern, compile as _regex_compile

TemplateParameter: TypeAlias = int | str


class Template(Protocol):
    def render(mapping: dict[str, TemplateParameter]) -> Sequence[str]: ...


class _Template:
    def __init__(self, template: _pyTemplate) -> None:
        self._template = template

    def render(self, mapping: dict[str, TemplateParameter]) -> str:
        """Replace all parameters in the template by the values provided by mapping.

        Will raise a `KeyError` if not all parameter values are defined in mapping.
        """
        return self._template.substitute(mapping)


@runtime_checkable
class TemplateParameterType(Protocol):
    regex: str

    def replace(self, m: Match[AnyStr]) -> str: ...


@runtime_checkable
class AnalysingTemplateParameterType(TemplateParameterType, Protocol):
    analyse_regex: str

    def analyse(self, m: Match) -> None: ...


class TemplateBuilder:
    """Builds a template based on a given prototype.

    The builder takes a prototype as a string and creates a template from it,
    based on specified template parameter types.
    A template parameter type specifies how a pattern inside the prototype shall
    be converted into a template parameter.

    Two different kinds of template parameter types are supported:

    `TemplateParameterType`:: will use a regular expression `.regex` to search for
        occurences of a pattern and replace them by the result of calling `.replace`
        on the found match. Usually the new value is a template parameter, e.g.,
        `"$my_template_param"`.
        NOTE: in case of multiple overlapping parameter types the first one added
        to the builder will take precedence.
    `AnalysingTemplateParameterType`:: works like `TemplateParameterType` but will
        first go through an analysing phase. This is useful to let the regular expression
        depend on the content of the prototype. E.g., we could find the first defined
        class in a python file and replace all occurences of that class name with
        a template parameter.

    Once build, the template is cached. The cache is invalidated as soon as
    new parameter types are added or the underlying prototype is changed.
    """

    def __init__(self) -> None:
        self._prototype = ""
        self._parameters: dict[str, TemplateParameterType] = dict()
        self._template = _pyTemplate("")
        self._cached_template_is_valid = False

    def set_prototype(
        self, prototype: str | tuple[str, ...] | list[str]
    ) -> "TemplateBuilder":
        self._invalidate_cache()
        if isinstance(prototype, str):
            self._prototype = prototype
        else:
            self._prototype = "\n".join(prototype)
        return self

    def add_parameter(self, name, _type: TemplateParameterType) -> "TemplateBuilder":
        self._invalidate_cache()
        self._parameters[name] = _type

    def build(self) -> Template:
        if not self._cached_template_is_valid:
            self._analyse()
            regex = self._build_regex()
            self._template = _pyTemplate(regex.sub(self._replace, self._prototype))
        return _Template(self._template)

    def _replace(self, m: Match) -> str:
        type_name = m.lastgroup
        replacement = self._parameters[type_name].replace
        return replacement(m)

    def _analyse(self) -> None:
        regex = self._build_analyse_regex()
        for match in regex.finditer(self._prototype):
            parameter = self._parameters[match.lastgroup]
            parameter.analyse(match)

    def _build_analyse_regex(self) -> Pattern:
        regex = "|".join(
            _type.analyse_regex.format(value=name)
            for name, _type in self._parameters.items()
            if isinstance(_type, AnalysingTemplateParameterType)
        )
        return _regex_compile(regex)

    def _build_regex(self) -> Pattern:
        regex = "|".join(
            _type.regex.format(value=name) for name, _type in self._parameters.items()
        )
        return _regex_compile(regex)

    def _invalidate_cache(self) -> None:
        self._cached_template_is_valid = False
