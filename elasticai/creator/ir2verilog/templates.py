from string import Template as _pyTemplate
from typing import Mapping, Self, cast

from elasticai.creator import template as tpl


class _Definition(tpl.TemplateParameter):
    def __init__(self, type: str, name: str, delimiter: str) -> None:
        self.name = name
        self.delimiter = delimiter
        self.regex = r"(?P<param>{type}\s+(signed\s+)?(\[.*?\]\s+)?{name}\s*=)\s?.[^;,\n]*".format(
            type=type, name=name
        )

    def replace(self, m: dict[str, str]) -> str:
        d = self.delimiter
        return f"{m['param']} {d}{self.name}"


class _LocalParameter(_Definition):
    """Turn the localparam `name` into a template parameter."""

    def __init__(self, name: str, delimiter: str):
        super().__init__("localparam", name, delimiter)


class _Parameter(_Definition):
    def __init__(self, name: str, delimiter: str):
        super().__init__("parameter", name, delimiter)


class _IdParameter(tpl.TemplateParameter):
    def __init__(self, name: str, delimiter: str):
        self.name = name
        self.delimiter = delimiter
        self.regex = r"(?P<prefix>[^_a-zA-Z0-9]*)(?P<id>{name})(?P<suffix>[^_a-zA-Z0-9]*)".format(
            name=name
        )

    def replace(self, m: dict[str, str]) -> str:
        d = self.delimiter
        return f"{m['prefix']}{d}{self.name}{m['suffix']}"


class _DefineSwitch(tpl.TemplateParameter):
    def __init__(self, name: str, delimiter: str):
        self.name = name
        self.regex = r"`define\s+{name}()(?=\s)".format(name=name)
        self.delimiter = delimiter

    def replace(self, m: dict[str, str]) -> str:
        d = self.delimiter
        return f"{d}def{self.name}"


class _ModuleName(tpl.AnalysingTemplateParameter):
    def __init__(self, delimiter: str):
        self._pre_analysis_regex = (
            r"(?P<prefix>[^_a-zA-Z0-9]){name}(?P<suffix>[^_a-zA-Z0-9])"
        )
        self.regex = ""
        self.analyse_regex = r"module\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
        self._analysed = False
        self._delimiter = delimiter

    def analyse(self, match: dict[str, str]) -> None:
        self._analysed = True
        name = match["name"]
        self.regex = self._pre_analysis_regex.format(name=name)

    def replace(self, match: dict[str, str]) -> str:
        d = self._delimiter
        if not self._analysed:
            raise ValueError("No module name found")
        return f"{match['prefix']}{d}module_name{match['suffix']}"


class TemplateDirector:
    """Director for verilog templates.

    Most methods correspond to verilog language constructs."""

    def __init__(self) -> None:
        self._builder = tpl.TemplateBuilder()
        self._define_switches: dict[str, bool] = {}
        self._delimiter = "§"

    def reset(self) -> Self:
        self._builder = tpl.TemplateBuilder()
        self._define_switches = {}
        return self

    def set_prototype(self, prototype: str) -> Self:
        self._builder.set_prototype(prototype)
        return self

    def parameter(self, name: str) -> Self:
        self._builder.add_parameter(_Parameter(name, self._delimiter))
        return self

    def localparam(self, name: str) -> Self:
        self._builder.add_parameter(_LocalParameter(name, self._delimiter))
        return self

    def define_scoped_switch(self, name: str, default=True) -> Self:
        """Add a switch for a define that is scoped to the module name.

        The switch will be prefixed with the value that users provide as `module_name` to the render call.
        """
        switch = _DefineSwitch(name, self._delimiter)
        self._builder.add_parameter(switch)
        self._builder.add_parameter(_IdParameter(name, self._delimiter))
        self._define_switches[name] = default
        return self

    def add_module_name(self) -> Self:
        self._builder.add_parameter(_ModuleName(self._delimiter))
        return self

    def build(self) -> "VerilogTemplate":
        MyStrTemplate = type(
            "MyStrTemplate", (_pyTemplate,), {"delimiter": self._delimiter}
        )
        return VerilogTemplate(
            MyStrTemplate(self._builder.build()), self._define_switches
        )


class VerilogTemplate:
    def __init__(self, template: _pyTemplate, defines: dict[str, bool]) -> None:
        self._template = template
        self._defines = defines.copy()

    def undef(self, name: str):
        """Disable a define switch, ie. comment out the define statement"""
        self._defines[name] = False

    def define(self, name: str):
        """Enable a define switch, ie. uncomment the define statement"""
        self._defines[name] = True

    def substitute(self, params: Mapping[str, str | bool]) -> str:
        new_params: dict[str, str] = {}
        if "module_name" in params:
            module_prefix = f"{params['module_name']}_"
        else:
            module_prefix = ""

        for name, value in params.items():
            if name not in self._defines:
                new_params[name] = cast(str, value)

        defines: dict[str, bool] = self._defines.copy()
        for name, enabled in params.items():
            if name in self._defines and isinstance(enabled, bool):
                defines[name] = enabled

        for name, enabled in defines.items():
            defname = f"def{name}"
            new_params[defname] = f"`define {module_prefix}{name}"
            new_params[name] = f"{module_prefix}{name}"
            if not enabled:
                new_params[defname] = "// automatically disabled\n// " + cast(
                    str, new_params[defname]
                )
        return self._template.substitute(new_params)
