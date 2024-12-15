import importlib.resources as res
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib import import_module
from inspect import signature
from typing import Generic, Protocol, TypeAlias, TypeVar

import tomlkit as toml


@dataclass
class Plugin:
    name: str
    target_platform: str
    target_runtime: str
    version: str
    api_version: str
    package: str


_PluginDict: TypeAlias = dict[str, str | list[str]]


PluginT = TypeVar("PluginT", bound=Plugin)


def read_plugins_from_package(p: str, plugin_type: type[PluginT]) -> list[PluginT]:
    def load_dicts_from_toml(toml_name) -> list[_PluginDict]:
        t = res.files(p).joinpath(f"{toml_name}.toml")
        parsed: list[_PluginDict] = []
        if t.is_file():
            with t.open("rb") as f:
                content = toml.load(f).unwrap()
                parsed.extend(content["plugins"])
        return parsed

    def build_plugin(d: _PluginDict) -> PluginT:
        args = {k: tuple(v) if not isinstance(v, str) else v for k, v in d.items()}
        s = signature(plugin_type)
        args["package"] = p
        bound = s.bind(**args)
        return plugin_type(*bound.args, **bound.kwargs)

    return list(map(build_plugin, load_dicts_from_toml("meta")))


Loader = TypeVar("Self", bound="BasePluginLoader")


class Loadable(Protocol[Loader]):
    @abstractmethod
    def load(self, loader: Loader) -> None: ...


class BasePluginLoader(ABC, Generic[Loader, PluginT]):
    """
    PluginLoader is responsible for dynamically searching for and loading plugins.
    It will find loadable plugins and call the `load` function with their expected
    arguments.
    """

    def __init__(self: Loader, *args, **kwargs):
        self._plugin_type: type[PluginT] = kwargs.pop("plugin_type")
        super().__init__(*args, **kwargs)

    def load_from_package(self: Loader, p: str) -> None:
        plugins = read_plugins_from_package(p, self._plugin_type)
        for _p in plugins:
            self.load(_p)

    @abstractmethod
    def _get_loadables(self: Loader, p: PluginT) -> dict[str, set[str]]:
        """
        Returns a dictionary with the modules containing the loadable objects as keys and
        a set of names of the loadable objects as values.
        """
        ...

    def load(self: Loader, p: PluginT) -> None:
        for m_name, components in self._get_loadables(p).items():
            module = import_module(m_name)
            for c_name in components:
                c: Loadable = getattr(module, c_name)
                c.load(self)
