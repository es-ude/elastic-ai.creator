import importlib.resources as res
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Protocol

import tomlkit as toml

from elasticai.creator.ir import Lowerable
from elasticai.creator.ir import LoweringPass as _LoweringPass

type _Attribute = (
    str
    | int
    | float
    | list[_Attribute]
    | tuple[_Attribute, ...]
    | dict[str, _Attribute]
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
    package: str


type _PluginDict = dict[str, str | list[str]]


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


type FN[_Tin, _Tout] = Callable[[_Tin], _Tout] | Callable[[_Tin], Iterable[_Tout]]


class PluggableTypeHandler[Tin: Lowerable, Tout](Protocol):
    @abstractmethod
    def load(self, loader: "PluginLoader[Tin, Tout]") -> None: ...


@dataclass(frozen=True)
class _SubFolderStructure:
    generated: str
    templates: str
    static_files: str


class PluginLoader[Tin: Lowerable, Tout]:
    folders = _SubFolderStructure(
        generated="src", templates="vhdl", static_files="static"
    )

    def __init__(self, lowering: _LoweringPass[Tin, Tout]):
        self._lowering = lowering

    @staticmethod
    def _get_from_module(package, module):
        import_module(package, module)

    def load(self, p: Plugin) -> None:
        package = f"{p.package}.{self.folders.generated}"
        for name in p.generated:
            m = import_module(package)
            h: PluggableTypeHandler = getattr(m, name)
            self.load_generated(h)

    def load_generated(self, h: PluggableTypeHandler) -> None:
        h.load(self)

    def register_iterable(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._lowering.register_iterable(name)(fn)

    def register(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._lowering.register(name)(fn)
