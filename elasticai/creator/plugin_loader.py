from abc import abstractmethod
from dataclasses import dataclass
from importlib import import_module
from importlib import resources as res
from inspect import signature
from typing import Callable, Iterable, Protocol, TypeAlias, TypeVar, Generic

import tomlkit as toml

from elasticai.creator.ir import Lowerable
from elasticai.creator.ir import LoweringPass as _LoweringPass
from elasticai.creator.plugin import Plugin


@dataclass(frozen=True)
class _SubFolderStructure:
    generated: str
    templates: str
    static_files: str


_PluginDict: TypeAlias = dict[str, str | list[str]]
Tin = TypeVar("Tin", bound=Lowerable)
Tout = TypeVar("Tout")


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


class PluggableTypeHandler(Protocol):
    @abstractmethod
    def load(self, loader: "PluginLoader[Tin, Tout]") -> None: ...


class PluginLoader(Generic[Tin, Tout]):
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

    def load_from_package(self, p: str) -> None:
        plugins = read_plugins_from_package(p)
        for _p in plugins:
            self.load(_p)
