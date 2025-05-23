"""Provides ways to use and extend the elasticai.creator plugin system.

The plugin systems evolves around the

* [`PluginLoader`](#elasticai.creator.plugin.PluginLoader) class
* [`PluginSymbol`](#elasticai.creator.plugin.PluginSymbol)protocol
* `meta.toml` files that describe the plugins

For convenience many functions convert the dicts loaded from the `meta.toml` file into `PluginSpec` objects.
The `meta.toml` file needs to define the value of each field of the `PluginSpec` class.
The function [`read_plugin_dicts_from_package`](#elasticai.creator.plugin.read_plugin_dicts_from_package) will read all plugins from the `plugins` key in the `meta.toml` file of a package.

**Example of a minimal `meta.toml` file**
```toml
[[plugins]]
name = "minimal_plugin"
target_platform = "elastic-node-v5"
target_runtime = "vhdl"
version = "0.1"
api_version = "0.1"
```

The few minimal fields that a plugin is required to define shall allow
plugin loaders to decide how to treat the plugin this could mean to

* ignore the plugin
* forward it to another software component
* check if the plugin is not compatible with the current setup

The following table lists these required fields:

:::{list-table}
* - Field name
  - Type
  - Description
* - **name**
  - `str`
  - The name of the plugin, used to identify the plugin
* - **target platform**
  - `str`
  - A string describing the target platform for the plugin, ie.
    the lowering pass it should be loaded into.
    Currently there is no strict definition of the semantics of this string.
* - **target runtime**
  - `str`
  - A string the runtime context for the plugin.
    Currently there is no strict definition of the semantics of this string.
* - **version**
  - `str`
  - A version string in the form `major.minor.[patch]`.
    Specifies the version of the plugin, ie. if you introduce a new feature or fix a bug, you should usually increase the minor version.
* - **api_version**
  - `str`
  - The version of the plugin API (plugin system) that this plugin was developed against.
    This is used to check if the plugin is compatible with the current system.
:::


:::{warning}
The set of required fields and their semantics is experimental and likely to change in the future.
:::

The `PluginLoader` will read that description from the `meta.toml` file
in a given package and use a user provided function to decide which
symbols to load from which module.
Assuming that each of these symbols implements the `PluginSymbol` protocol,
it will then call `load_into` on each of these symbols with a `plugin_receiver`.
The `plugin_receiver` is provided by the user as well.

Most other classes defined in this module are supposed to increase usability and expressiveness.

"""

import importlib.resources as _res
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial, update_wrapper
from importlib import import_module as _import_module
from inspect import signature as _signature
from typing import Generic, ParamSpec, Protocol, TypeAlias, TypeVar, Union

import tomlkit as toml


@dataclass
class PluginSpec:
    """The specification of a plugin.

    Typically built by reading a dictionary from a toml file and
    building the spec using <<build_plugin_spec, `build_plugin_spec()`>>.
    The dataclass is only used to provide convenient access to the fields,
    support type checking and improve code readability.

    You can achieve your goals just as well with the `PluginDict` dictionary.
    That is defined as an alias for `dict[str, str | tuple[str, ...]]`.
    """

    name: str
    target_platform: str
    target_runtime: str
    version: str
    api_version: str
    package: str


PluginDict: TypeAlias = dict[str, Union[str, tuple[str, ...], "PluginDict"]]

_PlRecT = TypeVar("_PlRecT", contravariant=True)
_T = TypeVar("_T")


PluginSpecT = TypeVar("PluginSpecT", bound=PluginSpec)

P = ParamSpec("P")

_ReturnT = TypeVar("_ReturnT", covariant=True)


class PluginLoader(Generic[_PlRecT]):
    """Get a set of plugins from a package, use `extract_fn` to resolve the symbols and load them into the `plugin_receiver`.

    .Args
    * `fetch`: A function that extracts ``PluginSymbol``s from plugin dictionaries. Use this to decide, based on the plugin specs, which symbols to load from which modules. Use the `SymbolFetcherBuilder` to easily compose new `fetch` functions.

    * `plugin_receiver`: The object that will receive the loaded symbols. _Receive_ in this context means that the loader will call `symbol.load_into(plugin_receiver)` for each of these symbols.
    That allows symbols defined in the plugin to run code in the context of the plugin receiver.
    E.g., it could register a lowering function in a `LoweringPass` if that lowering pass is given as the receiver.
    """

    def __init__(self, fetch: "SymbolFetcher", plugin_receiver: _PlRecT) -> None:
        self._extract_fn = fetch
        self._receiver = plugin_receiver

    def _get_plugin_dicts(self, package: str) -> Iterable[PluginDict]:
        yield from read_plugin_dicts_from_package(package)

    def load_from_package(self, package: str) -> None:
        """load all plugins defined by the `meta.toml` file found in the package."""
        plugin_dicts = self._get_plugin_dicts(package)
        symbols = self._extract_fn(plugin_dicts)
        for loadable in symbols:
            loadable.load_into(self._receiver)


class PluginSymbol(Protocol[_PlRecT]):
    """A symbol that the `PluginLoader` can load into a receiver object.

    The receiver can be any object.
    """

    @abstractmethod
    def load_into(self, /, receiver: _PlRecT) -> None: ...


class SymbolFetcher(Protocol[_PlRecT]):
    """Fetches ``PluginSymbol``s for the `PluginLoader`."""

    @abstractmethod
    def __call__(
        self, data: Iterable[PluginDict]
    ) -> Iterator[PluginSymbol[_PlRecT]]: ...


_SymbolFetcherBuilderT = TypeVar("_SymbolFetcherBuilderT", bound="SymbolFetcherBuilder")


class SymbolFetcherBuilder(Generic[PluginSpecT, _PlRecT]):
    """Build a `SymbolFetcher` from simpler functions.

    The `SymbolFetcherBuilder` composes simpler functions into a
    `SymbolFetcher`.

    .Args
    * `spec_type`: The type of the plugin spec that the `SymbolFetcher` will build from the plugin dictionaries.
    """

    def __init__(self, spec_type: type[PluginSpecT]) -> None:
        self._fns_over_iterables: list[
            Callable[[Iterable[PluginSpecT]], Iterator[PluginSymbol[_PlRecT]]]
        ] = []
        self._fns: list[Callable[[PluginSpecT], Iterator[PluginSymbol[_PlRecT]]]] = []
        self._spec_type: type[PluginSpecT] = spec_type

    def add_fn_over_iter(
        self: _SymbolFetcherBuilderT,
        fn: Callable[[Iterable[PluginSpecT]], Iterator[PluginSymbol[_PlRecT]]],
    ) -> _SymbolFetcherBuilderT:
        """Add a function that will be called for each plugin spec."""
        self._fns_over_iterables.append(fn)
        return self

    def add_fn(
        self: _SymbolFetcherBuilderT,
        fn: Callable[[PluginSpecT], Iterator[PluginSymbol[_PlRecT]]],
    ) -> _SymbolFetcherBuilderT:
        """Add a function that will be called once for all plugin specs."""
        self._fns.append(fn)
        return self

    def build(self) -> SymbolFetcher[_PlRecT]:
        # The next three locs will decouple `fetcher` from the builder's state.
        # Without them the function will keep looking up these values
        # from the builder's namespace instead of the function closure.
        # But the builders state might have changed in the meantime.
        spec_type = self._spec_type
        fns = self._fns
        fns_over_iterables = self._fns_over_iterables

        def fetcher(data: Iterable[PluginDict]) -> Iterator[PluginSymbol[_PlRecT]]:
            specs = map(partial(build_plugin_spec, spec_type=spec_type), data)
            for plugin in specs:
                for fn in fns:
                    yield from fn(plugin)

            for iter_fn in fns_over_iterables:
                yield from iter_fn(specs)

        return fetcher


class PluginSymbolFn(PluginSymbol[_PlRecT], Generic[_PlRecT, P, _ReturnT], Protocol):
    """A `PluginSymbol` that is also a function/callable."""

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> _ReturnT: ...


def import_symbols(module: str, names: Iterable[str]) -> Iterator[PluginSymbol]:
    """import names from a module and yield the resulting objects."""
    m = _import_module(module)
    for name in set(names):
        yield getattr(m, name)


def make_plugin_symbol(
    load_into: Callable[[_PlRecT], None], fn: Callable[P, _T]
) -> PluginSymbolFn[_PlRecT, P, _T]:
    """Turn two functions into a loadable and callable plugin symbol.

    .Args
    * `load_into`: executed when `PluginLoader` calls `load_into` on the Symbol
    * `fn`: wrapped function. Executed when the created `PluginSymbolFn` is called.

    An important use case for plugin symbols is to provide callable functions,
    e.g., for lowering passes. `make_plugin_symbol` eases creating
    plugin symbols for these functions.

    For an example look at the implementation of the `type_handler`
    decorators in xref::api:ir2vhdl.adoc[ir2vhdl].
    ```
    """

    class _PS(PluginSymbolFn[_PlRecT, P, _T]):
        def load_into(self, receiver):
            load_into(receiver)

        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> _T:
            return fn(*args, **kwargs)

    wrapped = _PS()

    update_wrapper(wrapped, fn)
    return wrapped


def build_plugin_spec(d: PluginDict, spec_type: type[PluginSpecT]) -> PluginSpecT:
    """inspect spec_type and build an instance of it from the dictionary `d`.

    Missing field raise an error while extra fields will be ignored.
    """
    args = d
    s = _signature(spec_type)
    expected_params = set(s.parameters.keys())
    actual_params = set(args.keys())
    if expected_params != actual_params:
        if actual_params.intersection(expected_params).issubset(expected_params):
            raise MissingFieldError(
                expected_params.difference(actual_params), spec_type
            )
    bound = {k: args[k] for k in s.parameters.keys()}
    return spec_type(**bound)  # type: ignore


def read_plugin_dicts_from_package(package: str) -> Iterable[PluginDict]:
    """read the meta.toml file from the package returning the list of plugin dictionaries."""
    t = _res.files(package).joinpath("meta.toml")
    parsed: list[PluginDict] = []
    if t.is_file():
        with t.open("rb") as f:
            content = toml.load(f).unwrap()
            parsed.extend(content["plugins"])
    for d in parsed:
        d.update(dict(package=package))
    return parsed


class MissingFieldError(Exception):
    def __init__(self, field_names: set[str], plugin_type: type[PluginSpecT]):
        super().__init__(
            f"missing required fields {field_names} for plugin spec '{plugin_type.__qualname__}'\n\tAre you sure you are loading the correct plugin?\n\tIs the meta.toml file correct?"
        )
