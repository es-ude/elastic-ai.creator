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


PluginDict: TypeAlias = dict[str, str | list[str]]


PluginT = TypeVar("PluginT", bound=Plugin)


def build_plugin(d: PluginDict, plugin_type: type[PluginT]) -> PluginT:
    """
    raises an `UnexpectedFieldError` in case `d` contains unexpected fields
    """
    args = {k: tuple(v) if not isinstance(v, str) else v for k, v in d.items()}
    s = signature(plugin_type)
    expected_params = set(s.parameters.keys())
    actual_params = set(args.keys())
    if not expected_params == actual_params:
        if expected_params.issubset(actual_params):
            raise UnexpectedFieldError(
                actual_params.difference(expected_params), plugin_type
            )
    bound = s.bind(**args)
    return plugin_type(*bound.args, **bound.kwargs)


def read_plugins_from_package(p: str, plugin_type: type[PluginT]) -> list[PluginT]:
    """
    raises an `UnexpectedFieldError` in case any discovered config contains unexpected fields
    """

    def load_dicts_from_toml(toml_name) -> list[PluginDict]:
        t = res.files(p).joinpath(f"{toml_name}.toml")
        parsed: list[PluginDict] = []
        if t.is_file():
            with t.open("rb") as f:
                content = toml.load(f).unwrap()
                parsed.extend(content["plugins"])
        for d in parsed:
            d.update(dict(package=p))
        return parsed

    def _build(d: PluginDict) -> PluginT:
        return build_plugin(d, plugin_type)

    return list(map(_build, load_dicts_from_toml("meta")))


Loader = TypeVar("Loader", bound="BasePluginLoader", contravariant=True)


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
        Returns a dictionary, it's keys are the modules that contain the loadable objects,
        the values are sets of names of the objects we will want to load from these modules.
        """
        ...

    def _post_load_hook(self: Loader) -> None:
        pass

    def _pre_load_hook(self: Loader) -> None:
        pass

    def load(self: Loader, p: PluginT) -> None:
        self._pre_load_hook()
        for m_name, components in self._get_loadables(p).items():
            module = import_module(m_name)
            for c_name in components:
                c: Loadable = getattr(module, c_name)
                c.load(self)
        self._post_load_hook()


class UnexpectedFieldError(Exception):
    def __init__(self, field_names: set[str], plugin_type: type[PluginT]):
        super().__init__(
            f"unexpected fields {field_names} for plugin '{plugin_type.__qualname__}'"  # type: ignore
        )
