import re
from collections.abc import Iterable
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
        ps.build_plugin_spec(config_from_file, p.PluginSpec)  # type: ignore


class DummySymbol:
    loaded = False


class MinimalPluginLoader(p.PluginLoaderBase[p.PluginSpec, DummySymbol]):
    def __init__(self):
        super().__init__(p.PluginSpec)
        self.symbols: list[DummySymbol] = []

    def filter_plugin_dicts(
        self, plugins: Iterable[dict[str, Any]]
    ) -> Iterable[dict[str, Any]]:
        return plugins

    def get_symbols(self, specs: Iterable[p.PluginSpec]) -> Iterable[DummySymbol]:
        for symbol in p.import_symbols("tests.unit_tests.plugin_test", {"DummySymbol"}):
            yield symbol

    def load_symbol(self, symbol: DummySymbol) -> None:
        symbol.loaded = True
        self.symbols.append(symbol)


def test_plugin_loader_loads_plugins():
    DummySymbol.loaded = False
    loader = MinimalPluginLoader()
    test_spec = p.PluginSpec(
        name="test",
        version="0.1",
        api_version="0.1",
        package="tests.unit_tests.plugin_test",
        target_platform="test",
        target_runtime="test",
    )
    symbols = list(loader.get_symbols([test_spec]))
    for symbol in symbols:
        loader.load_symbol(symbol)
    assert DummySymbol.loaded
