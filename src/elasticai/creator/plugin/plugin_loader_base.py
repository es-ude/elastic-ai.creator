import dataclasses
from abc import abstractmethod
from collections.abc import Iterable
from importlib import resources as res

from .plugin_spec import PluginMap, PluginSpec
from .read_specs_from_toml import read_plugin_dicts_from_package


class PluginLoaderBase[PS: PluginSpec, S]:
    """PluginLoader for translation passes."""

    def __init__(self, spec: type[PS]):
        self._spec: type[PS] = spec

    def load_from_package(self, package: str) -> None:
        if "." not in package:
            package = f"elasticai.creator_plugins.{package}"

        for symbol in self.get_symbols(self.get_specs(package)):
            self.load_symbol(symbol)

    @abstractmethod
    def filter_plugin_dicts(
        self, plugins: Iterable[PluginMap]
    ) -> Iterable[PluginMap]: ...

    @abstractmethod
    def load_symbol(self, symbol: S) -> None: ...

    @abstractmethod
    def get_symbols(self, specs: Iterable[PS]) -> Iterable[S]: ...

    def get_specs(self, package: res.Anchor) -> Iterable[PS]:
        spec_fields = set(f.name for f in dataclasses.fields(self._spec))
        plugin_dicts: Iterable[PluginMap] = self.filter_plugin_dicts(
            read_plugin_dicts_from_package(package)
        )
        plugin_dicts = ({k: p[k] for k in p if k in spec_fields} for p in plugin_dicts)
        plugin_specs = (self._spec(**p) for p in plugin_dicts)  # type: ignore
        return plugin_specs


class StaticFileBase:
    def __init__(self, name: str, package: str, subfolder: str):
        self._name = name
        self._package = package
        self._subfolder = subfolder

    @property
    def name(self) -> str:
        return self._name

    def get_content(self) -> str:
        file = res.files(self._package).joinpath(f"{self._subfolder}/{self.name}")
        return file.read_text()
