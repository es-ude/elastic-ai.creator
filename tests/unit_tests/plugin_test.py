from abc import abstractmethod
from collections.abc import Iterable, Iterator
from importlib import import_module
from typing import Generic, NamedTuple, Protocol, TypeVar, cast

import pytest

import elasticai.creator.plugin as p


def test_importing_plugin_with_unkown_fields_raises_meaningful_error():
    config_from_file = {
        "name": "my_plugin",
        "api_version": "0.1",
        "version": "0.2",
        "package": "mypackage",
        "target_platform": "env5",
        "target_runtime": "vhdl",
        "new_unknown_field": "value",
    }
    with pytest.raises(
        p.UnexpectedFieldError,
        match="unexpected fields {'new_unknown_field'} for plugin spec 'PluginSpec'",
    ):
        p.build_plugin_spec(config_from_file, p.PluginSpec)


class MinimalPluginLoader(p.PluginLoader):
    def __init__(self, extract_fn: p.SymbolFetcher):
        super().__init__(extract_fn, self)

    def _get_plugin_dicts(self, package: str) -> Iterable[p.PluginDict]:
        dummy_dicts: list[p.PluginDict] = []
        return dummy_dicts


class DummyLoadable(p.PluginSymbol[p.PluginLoader]):
    loaded = False

    @classmethod
    def load_into(cls, loader: p.PluginLoader) -> None:
        cls.loaded = True


def test_plugin_loader_loads_plugins():
    def extract_fn(data: Iterable[p.PluginDict]) -> Iterator[p.PluginSymbol]:
        yield from p.import_symbols("tests.unit_tests.plugin_test", {"DummyLoadable"})

    DummyLoadable.loaded = False
    loader = MinimalPluginLoader(extract_fn)
    loader.load_from_package("")
    assert DummyLoadable.loaded
