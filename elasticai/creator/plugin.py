import importlib.resources as res
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Protocol, TypeAlias

import tomlkit as toml

from elasticai.creator.ir import Lowerable
from elasticai.creator.ir import LoweringPass as _LoweringPass


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
    package: str


_PluginDict: TypeAlias = dict[str, str | list[str]]


def read_plugins_from_package(p: str) -> list[Plugin]:
    def load_dicts_from_toml(toml_name) -> list[_PluginDict]:
        t = res.files(p).joinpath(f"{toml_name}.toml")
        parsed: list[_PluginDict] = []
        if t.is_file():
            with t.open("rb") as f:
                content = toml.load(f).unwrap()
                parsed.extend(content["plugins"])
        return parsed

    def build_plugin(d: _PluginDict) -> Plugin:
        args = {k: tuple(v) if not isinstance(v, str) else v for k, v in d.items()}
        s = signature(Plugin)
        args["package"] = p
        bound = s.bind(**args)
        return Plugin(*bound.args, **bound.kwargs)

    return list(map(build_plugin, load_dicts_from_toml("meta")))
