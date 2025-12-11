from .loading import PluginLoader as _PluginLoader
from .loading import SymbolFetcher, SymbolFetcherBuilder, import_symbols
from .plugin_spec import MissingFieldError, PluginMap, PluginSpec, build_plugin_spec
from .read_specs_from_toml import (
    read_plugin_dicts_from_package as _read_plugin_dicts_from_package,
)
from .symbol import PluginSymbol, PluginSymbolFn
from .symbol_provider import make_plugin_symbol

__all__ = [
    "import_symbols",
    "make_plugin_symbol",
    "build_plugin_spec",
    "PluginLoader",
    "PluginSymbol",
    "PluginSymbolFn",
    "SymbolFetcherBuilder",
    "PluginSpec",
    "PluginMap",
    "MissingFieldError",
    "SymbolFetcher",
]


class PluginLoader[RecT](_PluginLoader):
    def __init__(self, fetch: SymbolFetcher[RecT], plugin_receiver: RecT) -> None:
        super().__init__(
            fetch=fetch,
            plugin_receiver=plugin_receiver,
            read_plugin_dicts_from_package=_read_plugin_dicts_from_package,
        )
