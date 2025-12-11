import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import pytest

import elasticai.creator.plugin as p
import elasticai.creator.plugin.plugin_spec as ps


def test_importing_plugin_with_missing_fields_raises_meaningful_error() -> None:
    config_from_file = {
        "api_version": "0.1",
        "version": "0.2",
        "package": "mypackage",
        "target_platform": "env5",
        "target_runtime": "vhdl",
        "new_unknown_field": "value",
    }

    with pytest.raises(
        p.MissingFieldError,
        match=re.escape(
            "missing required fields {'name'} for plugin spec 'PluginSpec'\n\tAre you sure you are loading the correct plugin?\n\tIs the meta.toml file correct?"
        ),
    ):
        ps.build_plugin_spec(config_from_file, p.PluginSpec)


class MinimalPluginLoader(p.PluginLoader):
    def __init__(self, extract_fn: p.SymbolFetcher):
        super().__init__(extract_fn, self)

    def _get_plugin_dicts(self, package: str) -> Iterable[p.PluginMap]:
        dummy_dicts: list[p.PluginMap] = []
        return dummy_dicts


class DummyLoadable(p.PluginSymbol[p.PluginLoader]):
    loaded = False

    @classmethod
    def load_into(cls, loader: p.PluginLoader) -> None:
        cls.loaded = True


def test_call_symbol_fetcher_callbacks() -> None:
    class MySpec(p.PluginSpec):
        name: str

    class MySymbol(p.PluginSymbol):
        def load_into(self, receiver: Any) -> None:
            pass

    config_from_file = {
        "api_version": "0.1",
        "name": "my_plugin",
        "version": "0.2",
        "package": "mypackage",
        "target_platform": "env5",
        "target_runtime": "vhdl",
        "new_unknown_field": "value",
    }

    b: p.SymbolFetcherBuilder[MySpec, Any] = p.SymbolFetcherBuilder(MySpec)
    symbol = MySymbol()

    def fetch_callback(data: Iterable[MySpec]) -> Iterator[p.PluginSymbol]:
        yield symbol

    fetcher = b.add_fn_for_all(fetch_callback).build()
    symbols = list(fetcher([config_from_file]))
    assert symbols == [symbol]


def test_plugin_loader_loads_plugins():
    def extract_fn(data: Iterable[p.PluginMap]) -> Iterator[p.PluginSymbol]:
        yield from p.import_symbols("tests.unit_tests.plugin_test", {"DummyLoadable"})

    DummyLoadable.loaded = False
    loader = MinimalPluginLoader(extract_fn)
    loader.load_from_package("")
    assert DummyLoadable.loaded


def test_can_make_and_load_plugin_symbol() -> None:
    class MyReceiver:
        registered: list[Callable[[str], str]] = []

        def register(self, fn: Callable[[str], str]) -> None:
            self.registered.append(fn)

    def make_symbol(fn) -> p.PluginSymbol[MyReceiver]:
        def load_into(receiver: MyReceiver) -> None:
            receiver.register(fn)

        return p.make_plugin_symbol(load_into=load_into, fn=fn)

    @make_symbol
    def symbol(x: str) -> str:
        return "symbol_" + x

    receiver = MyReceiver()
    symbol.load_into(receiver)

    assert receiver.registered[0]("a") == "symbol_a"
