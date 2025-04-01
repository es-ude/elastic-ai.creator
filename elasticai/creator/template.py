from collections.abc import Callable
from functools import singledispatchmethod
from re import Match, Pattern
from re import compile as _regex_compile
from typing import Iterable, Protocol, runtime_checkable


@runtime_checkable
class TemplateParameter(Protocol):
    regex: str
    """A regular expression to match what should be turned into a template parameter."""

    def replace(self, match: dict[str, str]) -> str:
        """Define how to replace the matched pattern by the template parameter.

        The match is a dictionary of named capture groups.
        In the most simple cases you can just return `f"${self.name}"` to replace the
        full match by the template parameter.
        But you can also use the match to create a more complex replacement.

        The full match is available as `match[""]`.
        """
        ...


@runtime_checkable
class AnalysingTemplateParameter(TemplateParameter, Protocol):
    """A template parameter type that needs an analysis phase before the replacement.

    You could e.g., use this to find the first class definition in a python file and
    use that class name to build the `regex` parameter and the replacement to
    replace all occurences of that class name by a template parameter.

    Example:
    ```python
    class ClassNameParameter:
        name = "class_name"
        analyse_regex = r"class (?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"

        def __init__(self):
            self._class_name = None

        def analyse(self, match: dict[str, str]) -> None:
            # Store first found class name
            if self._class_name is None:
                self._class_name = match["name"]

        @property
        def regex(self) -> str:
            # After analysis, look for all occurrences of the class name
            return self._class_name

        def replace(self, match: dict[str, str]) -> str:
            # Replace with template parameter
            return f"${self.name}"

    # Usage:
    builder = TemplateBuilder()
    builder.set_prototype('''
        class MyClass:
            def method(self):
                return MyClass()
    ''')
    builder.add_parameter(ClassNameParameter())
    template = builder.build()
    print(template.render(dict(class_name="MyNewClass"))) # Will replace all occurrences of "MyClass" with "MyNewClass"
    ```
    """

    analyse_regex: str
    """Will be used during the analysis phase.
    
    See [`analyse`](#analyse) for more information.
    """

    def analyse(self, match: dict[str, str]) -> None:
        """Called for each match of `analyse_regex` during the analysis phase.

        :param match: A dictionary of the named capture groups.
        """
        ...


class TemplateBuilder:
    """Builds a template based on a given prototype.

    The builder takes a *prototype* as a string and creates a template from it,
    based on specified template parameter types.
    A template parameter type specifies how a pattern inside the prototype shall
    be converted into a template parameter.

    Two different kinds of template parameter types are supported:

    `TemplateParameterType`

    : will use a regular expression `.regex` to search for
        occurences of a pattern and replace them by the result of calling `.replace`
        on the found match. Usually the new value is a template parameter, e.g.,
        `"$my_template_param"`.
        :::{note}
        in case of multiple overlapping parameter types the first one added
        to the builder will take precedence.
        :::


    `AnalysingTemplateParameterType`

    :  works like `TemplateParameterType` but will
        first go through an analysing phase. This is useful to let the regular expression
        depend on the content of the prototype. E.g., we could find the first defined
        class in a python file and replace all occurences of that class name with
        a template parameter.

    Once build, the template is cached. The cache is invalidated as soon as
    new parameter types are added or the underlying prototype is changed.
    """

    def __init__(self) -> None:
        self._prototype = ""
        self._parameters: dict[str, _ParameterTypeWrapper] = dict()
        self._analysing_template_parameters: dict[str, _AnalyseParameterTypeWrapper] = (
            dict()
        )
        self._template = ""
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

    def add_parameter(
        self, _type: TemplateParameter | AnalysingTemplateParameter
    ) -> "TemplateBuilder":
        self._invalidate_cache()
        self._parameter_type_adder(_type)()
        return self

    @singledispatchmethod
    def _parameter_type_adder(
        self, _type: TemplateParameter | AnalysingTemplateParameter
    ) -> Callable[[], None]:
        raise NotImplementedError()

    @_parameter_type_adder.register
    def _(self, _type: TemplateParameter) -> Callable[[], None]:
        def adder():
            name = str(len(self._parameters))
            name = "_{name}".format(name=name)
            self._parameters[name] = _ParameterTypeWrapperImpl(name, _type)

        return adder

    @_parameter_type_adder.register
    def _(self, _type: AnalysingTemplateParameter) -> Callable[[], None]:
        def adder():
            name = str(len(self._parameters))
            name = "_{name}".format(name=name)
            wrapped = _AnalyseParameterTypeWrapper(name, _type)

            self._parameters[name] = wrapped
            self._analysing_template_parameters[name] = wrapped

        return adder

    def build(self) -> str:
        if not self._cached_template_is_valid:
            self._analyse()
            regex = self._build_replacement_regex()
            self._template = regex.sub(self._replace, self._prototype)
        return self._template

    def _replace(self, m: Match) -> str:
        type_name = m.lastgroup
        if type_name is not None:
            replacement = self._parameters[type_name].replace
        else:
            raise ValueError("no match found")
        return replacement(m)

    def _analyse(self) -> None:
        regex = self._build_analyse_regex()
        for match in regex.finditer(self._prototype):
            analyse = self._get_analyser(match.lastgroup)
            analyse(match)

    def _get_analyser(self, name: str | None) -> Callable[[Match], None]:
        if name is not None and name in self._analysing_template_parameters:
            return self._analysing_template_parameters[name].analyse
        else:

            def analyse(m: Match):
                pass

            return analyse

    def _build_analyse_regex(self) -> Pattern:
        return self._build_regex(
            (name, _type.analyse_regex)
            for name, _type in self._analysing_template_parameters.items()
        )

    def _build_replacement_regex(self) -> Pattern:
        return self._build_regex(
            (name, _type.regex) for name, _type in self._parameters.items()
        )

    def _build_regex(self, types: Iterable[tuple[str, str]]) -> Pattern:
        regex = "|".join(
            "(?P<{name}>{regex})".format(name=name, regex=type_regex)
            for name, type_regex in types
        )
        return _regex_compile(regex)

    def _invalidate_cache(self) -> None:
        self._cached_template_is_valid = False


class _ParameterTypeWrapper(Protocol):
    @property
    def regex(self) -> str: ...

    @property
    def name(self) -> str: ...

    def replace(self, m: Match) -> str: ...


class _ParameterTypeWrapperImpl:
    def __init__(self, name: str, _type: TemplateParameter) -> None:
        self._type = _type
        self.name = name

    @property
    def regex(self) -> str:
        return _mangle_capture_group_names(self.name, self._type.regex)

    def replace(self, m: Match) -> str:
        return self._type.replace(
            _demangle_capture_group_names(self.name, m.groupdict())
        )


class _AnalyseParameterTypeWrapper:
    def __init__(self, name: str, _type: AnalysingTemplateParameter) -> None:
        self._type = _type
        self.name = name

    @property
    def regex(self) -> str:
        return _mangle_capture_group_names(self.name, self._type.regex)

    @property
    def analyse_regex(self) -> str:
        return _mangle_capture_group_names(self.name, self._type.analyse_regex)

    def analyse(self, match: Match) -> None:
        self._type.analyse(_demangle_capture_group_names(self.name, match.groupdict()))

    def replace(self, match: Match) -> str:
        match_dict = _demangle_capture_group_names(self.name, match.groupdict())
        return self._type.replace(match_dict)


def _mangle_capture_group_names(name, regex: str) -> str:
    capture_group_pattern = r"\(\?P<(?P<capture>[^>]+)>"

    def replace(match: Match) -> str:
        return f"(?P<{name}_{match['capture']}>"

    return _regex_compile(capture_group_pattern).sub(replace, regex)


def _demangle_capture_group_names(name, match: dict[str, str]) -> dict[str, str]:
    new_dict = {}
    for k, v in match.items():
        if k.startswith(f"{name}"):
            k = k.removeprefix(f"{name}")
            k = k.removeprefix("_")
            new_dict[k] = v
    return new_dict
