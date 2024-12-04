import importlib.resources as res
import tomllib as toml
from dataclasses import dataclass
from inspect import signature
from typing import TypeAlias

_Attribute: TypeAlias = (
    str
    | int
    | float
    | list["_Attribute"]
    | tuple["_Attribute", ...]
    | dict[str, "_Attribute"]
)


@dataclass
class Plugin:
    name: str
    target_platform: str
    target_runtime: str
    version: str
    api_version: str
    generated: tuple[str, ...]
    templates: tuple[str, ...]
    static_files: tuple[str, ...]


_PluginDict: TypeAlias = dict[str, str | list[str]]


def read_plugins_from_package(p: str) -> list[Plugin]:
    def load_dicts_from_toml(toml_name) -> list[_PluginDict]:
        t = res.files(p).joinpath(f"{toml_name}.toml")
        parsed: list[_PluginDict] = []
        if t.is_file():
            with t.open("rb") as f:
                content = toml.load(f)
                print(content)
                parsed.extend(content["plugins"])
        return parsed

    def build_plugin(d: _PluginDict) -> Plugin:
        args = {k: tuple(v) if not isinstance(v, str) else v for k, v in d.items()}
        s = signature(Plugin)
        bound = s.bind(**args)
        return Plugin(*bound.args, **bound.kwargs)

    return list(map(build_plugin, load_dicts_from_toml("meta")))
