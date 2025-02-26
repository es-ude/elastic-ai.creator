from typing import Mapping, Self, cast

from elasticai.creator import template as tpl


class _Definition(tpl.TemplateParameter):
    def __init__(self, type: str, name: str):
        self.name = name
        self.regex = r"(?P<param>{type}\s+(signed\s+)?(\[.*?\]\s+)?{name}\s*=)\s?.[^;,\n]*".format(
            type=type, name=name
        )

    def replace(self, m: dict[str, str]) -> str:
        return f"{m['param']} ${self.name}"


class _LocalParameter(_Definition):
    """Turn the localparam `name` into a template parameter."""

    def __init__(self, name: str):
        super().__init__("localparam", name)


class _Parameter(_Definition):
    def __init__(self, name: str):
        super().__init__("parameter", name)


class _IdParameter(tpl.TemplateParameter):
    def __init__(self, name: str):
        self.name = name
        self.regex = r"(?P<prefix>[^_a-zA-Z0-9]*)(?P<id>{name})(?P<suffix>[^_a-zA-Z0-9]*)".format(
            name=name
        )

    def replace(self, m: dict[str, str]) -> str:
        return f"{m['prefix']}${self.name}{m['suffix']}"


class _DefineSwitch(tpl.TemplateParameter):
    def __init__(self, name: str):
        self.name = name
        self.regex = r"`define\s+{name}()(?=\s)".format(name=name)

    def replace(self, m: dict[str, str]) -> str:
        return f"$def{self.name}"


class _ModuleName(tpl.AnalysingTemplateParameter):
    def __init__(self):
        self._pre_analysis_regex = (
            r"(?P<prefix>[^_a-zA-Z0-9]){name}(?P<suffix>[^_a-zA-Z0-9])"
        )
        self.regex = ""
        self.analyse_regex = r"module\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
        self._analysed = False

    def analyse(self, match: dict[str, str]) -> None:
        self._analysed = True
        name = match["name"]
        self.regex = self._pre_analysis_regex.format(name=name)

    def replace(self, match: dict[str, str]) -> str:
        if not self._analysed:
            raise ValueError("No module name found")
        return f"{match['prefix']}$module_name{match['suffix']}"


class TemplateDirector:
    """Director for verilog templates.

    Most methods correspond to verilog language constructs."""

    def __init__(self) -> None:
        self._builder = tpl.TemplateBuilder()
        self._define_switches: dict[str, bool] = {}

    def reset(self) -> Self:
        self._builder = tpl.TemplateBuilder()
        self._define_switches = {}
        return self

    def set_prototype(self, prototype: str) -> Self:
        self._builder.set_prototype(prototype)
        return self

    def parameter(self, name: str) -> Self:
        self._builder.add_parameter(_Parameter(name))
        return self

    def localparam(self, name: str) -> Self:
        self._builder.add_parameter(_LocalParameter(name))
        return self

    def define_scoped_switch(self, name: str, default=True) -> Self:
        """Add a switch for a define that is scoped to the module name.

        The switch will be prefixed with the value that users provide as `module_name` to the render call.
        """
        switch = _DefineSwitch(name)
        self._builder.add_parameter(switch)
        self._builder.add_parameter(_IdParameter(name))
        self._define_switches[name] = default
        return self

    def add_module_name(self) -> Self:
        self._builder.add_parameter(_ModuleName())
        return self

    def build(self) -> "VerilogTemplate":
        return VerilogTemplate(self._builder.build(), self._define_switches)


class VerilogTemplate:
    def __init__(self, template: tpl.Template, defines: dict[str, bool]) -> None:
        self._template = template
        self._defines = defines.copy()

    def undef(self, name: str):
        """Disable a define switch, ie. comment out the define statement"""
        self._defines[name] = False

    def define(self, name: str):
        """Enable a define switch, ie. uncomment the define statement"""
        self._defines[name] = True

    def render(self, params: Mapping[str, str | bool]) -> str:
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
        return self._template.render(new_params)
