import warnings
from collections.abc import Callable, Iterable, Iterator
from functools import partial
from importlib import import_module as _import_module
from typing import Self

from .plugin_spec import PluginMap, PluginSpec, build_plugin_spec
from .symbol import PluginSymbol

type SymbolFetcher[RecT] = Callable[[Iterable[PluginMap]], Iterator[PluginSymbol[RecT]]]


def import_symbols(module: str, names: Iterable[str]) -> Iterator[PluginSymbol]:
    """import names from a module and yield the resulting objects."""
    m = _import_module(module)
    for name in set(names):
        yield getattr(m, name)


class PluginLoader[RecT]:
    """Get a set of plugins from a package, use `extract_fn` to resolve the symbols and load them into the `plugin_receiver`.

    .Args
    * `fetch`: A function that extracts ``PluginSymbol``s from plugin dictionaries. Use this to decide, based on the plugin specs, which symbols to load from which modules. Use the `SymbolFetcherBuilder` to easily compose new `fetch` functions.

    * `plugin_receiver`: The object that will receive the loaded symbols. _Receive_ in this context means that the loader will call `symbol.load_into(plugin_receiver)` for each of these symbols.
    That allows symbols defined in the plugin to run code in the context of the plugin receiver.
    E.g., it could register a lowering function in a `LoweringPass` if that lowering pass is given as the receiver.
    """

    def __init__(
        self,
        fetch: "SymbolFetcher[RecT]",
        plugin_receiver: RecT,
        read_plugin_dicts_from_package: Callable[[str], Iterable[PluginMap]],
    ) -> None:
        self._extract_fn = fetch
        self._receiver = plugin_receiver
        self._read_plugin_dicts_from_package = read_plugin_dicts_from_package

    def _get_plugin_dicts(self, package: str) -> Iterable[PluginMap]:
        yield from self._read_plugin_dicts_from_package(package)

    def load_from_package(self, package: str) -> None:
        """load all plugins defined by the `meta.toml` file found in the package."""
        plugin_dicts = self._get_plugin_dicts(package)
        symbols = self._extract_fn(plugin_dicts)
        for loadable in symbols:
            loadable.load_into(self._receiver)


class SymbolFetcherBuilder[SpecT: PluginSpec, RecT]:
    """Build a `SymbolFetcher` from simpler functions.

    The `SymbolFetcherBuilder` composes simpler functions into a
    `SymbolFetcher`.

    .Args
    * `spec_type`: The type of the plugin spec that the `SymbolFetcher` will build from the plugin dictionaries.
    """

    def __init__(self, spec_type: type[SpecT]) -> None:
        self._fns_for_each: list[
            Callable[[Iterable[SpecT]], Iterator[PluginSymbol[RecT]]]
        ] = []
        self._fns: list[Callable[[SpecT], Iterator[PluginSymbol[RecT]]]] = []
        self._spec_type: type[SpecT] = spec_type

    def add_fn_for_all(
        self: Self,
        fn: Callable[[Iterable[SpecT]], Iterator[PluginSymbol[RecT]]],
    ) -> Self:
        """Add a function that will be called once for all plugin specs."""
        self._fns_for_each.append(fn)
        return self

    def add_fn_over_iter(
        self: Self,
        fn: Callable[[Iterable[SpecT]], Iterator[PluginSymbol[RecT]]],
    ) -> Self:
        """Add a function that will be called once for all plugin specs."""
        warnings.warn(
            "SymbolFetcherBuilder.add_fn_over_iter is deprecated, use add_fn_for_all instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self._fns_for_each.append(fn)
        return self

    def add_fn(
        self: Self,
        fn: Callable[[SpecT], Iterator[PluginSymbol[RecT]]],
    ) -> Self:
        """Add a function that will be called for each plugin spec."""
        self._fns.append(fn)
        return self

    def build(self) -> SymbolFetcher[RecT]:
        # The next three locs will decouple `fetcher` from the builder's state.
        # Without them the function will keep looking up these values
        # from the builder's namespace instead of the function closure.
        # But the builders state might have changed in the meantime.
        spec_type = self._spec_type
        fns = self._fns
        fns_over_iterables = self._fns_for_each

        def fetcher(data: Iterable[PluginMap]) -> Iterator[PluginSymbol[RecT]]:
            specs = map(partial(build_plugin_spec, spec_type=spec_type), data)
            for plugin in specs:
                for fn in fns:
                    yield from fn(plugin)

            for iter_fn in fns_over_iterables:
                yield from iter_fn(specs)

        return fetcher
