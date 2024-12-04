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


def read_plugin_from_package(p: str) -> Plugin:
    t = res.files(p).joinpath("meta.toml")
    if t.is_file():
        with t.open("rb") as f:
            parsed: dict[str, dict[str, str | list[str] | tuple[str]]] = toml.load(f)

    args = {
        k: tuple(v) if not isinstance(v, str) else v
        for k, v in parsed["plugin"].items()
    }

    s = signature(Plugin)
    if "name" not in args:
        args["name"] = res.files(p).name
    bound = s.bind(**args)
    return Plugin(*bound.args, **bound.kwargs)
