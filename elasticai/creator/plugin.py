"""Provides ways to use and extend the elasticai.creator plugin system.

The plugin systems evolves around the

* <<PluginLoader, `PluginLoader`>> class
* <<PluginSymbol, `PluginSymbol`>> protocol
* `meta.toml` files that describe the plugins

For convenience a plugin is described be the `PluginSpec` dataclass.
The `meta.toml` file needs to define the value of each field of the `PluginSpec` class.
The function <<read_plugin_dicts_from_package, `read_plugin_dicts_from_package`()>> will read all plugins from the `plugins` key in the `meta.toml` file of a package.

.Example of a minimal `meta.toml` file
[source,toml]
----
[[plugins]]
name = "minimal_plugin"
target_platform = "elastic-node-v5"
target_runtime = "vhdl"
version = "0.1"
api_version = "0.1"
----

The few minimal fields that a plugin is required to define shall allow
plugin loaders to decide how to treat the plugin this could mean to

* ignore the plugin
* forward it to another software component
* check if the plugin is not compatible with the current setup

The following table lists these required fields:

|===
| field name
| type
| description

| **name**
| `str`
| The name of the plugin, used to identify the plugin

| **target platform**
| `str`
| A string describing the target platform for the plugin.
Currently there is no strict definition of the semantics of this string.

| **target runtime**
| `str`
| A string the runtime context for the plugin.
Currently there is no strict definition of the semantics of this string.

| **version**
| `str`
| A version string in the form `major.minor.[patch]`. Specifies the
version of the plugin, ie. if you introduce a new feature or fix a bug,
you should usually increase the minor version.

| **api_version**
| `str`
| The version of the plugin API (plugin system) that this plugin was
developed against. This is used to check if the plugin is compatible
with the current system.
|===

The `PluginLoader` will read that description from the `meta.toml` file
in a given package and use a user provided function to decide which
symbols to load from which module.
Assuming that each of these symbols implements the `PluginSymbol` protocol,
it will then call `load_into` on each of these symbols with a `plugin_receiver`.
The `plugin_receiver` is provided by the user as well.

Most other classes defined in this module are supposed to increase usability and expressiveness.

"""

import importlib.resources as res
from abc import abstractmethod
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from inspect import signature
from typing import Generic, NamedTuple, Protocol, TypeAlias, TypeVar, cast

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


PluginDict: TypeAlias = dict[str, str | tuple[str, ...]]


PluginSpecT = TypeVar("PluginSpecT", bound=PluginSpec)


def build_plugin_spec(d: PluginDict, spec_type: type[PluginSpecT]) -> PluginSpecT:
    """inspect plugin_type and build an instance of it from the dictionary `d` raising an error in case of unexpected fields

    raises an `UnexpectedFieldError` in case `d` contains unexpected fields
    """
    args = {k: tuple(v) if not isinstance(v, str) else v for k, v in d.items()}
    s = signature(spec_type)
    expected_params = set(s.parameters.keys())
    actual_params = set(args.keys())
    if not expected_params == actual_params:
        if expected_params.issubset(actual_params):
            raise UnexpectedFieldError(
                actual_params.difference(expected_params), spec_type
            )
    bound = s.bind(**args)
    return spec_type(*bound.args, **bound.kwargs)


def read_plugin_dicts_from_package(package: str) -> Iterable[PluginDict]:
    """read the meta.toml file from the package returning the list of plugin dictionaries."""
    t = res.files(package=package).joinpath("meta.toml")
    parsed: list[PluginDict] = []
    if t.is_file():
        with t.open("rb") as f:
            content = toml.load(f).unwrap()
            parsed.extend(content["plugins"])
    for d in parsed:
        d.update(dict(package=package))
    return parsed


_Tcov = TypeVar("_Tcov", covariant=True)
_Tcontra = TypeVar("_Tcontra", contravariant=True)
_T = TypeVar("_T")


class PluginSymbol(Protocol[_Tcontra]):
    """A symbol that the `PluginLoader` can load into a receiver object."""

    @abstractmethod
    def load_into(self, /, receiver: _Tcontra) -> None: ...


class SymbolSpec(NamedTuple):
    """Convenience type to represent a module and a set of symbols we want to load from it."""

    module: str
    symbols: set[str]


class SymbolResolver(Protocol):
    """A function that extracts `SymbolSpec`s from plugin dictionaries.

    This is the essential function that users need to provide to the `PluginLoader`.
    Use this function to decide which symbols to load from which module.
    """

    def __call__(
        self, data: Iterable[PluginDict]
    ) -> Iterator[SymbolSpec | tuple[str, set[str]]]: ...


def make_symbol_resolver(
    fn: Callable[[Iterable[PluginSpecT]], Iterator[tuple[str, set[str]]]],
    plugin_spec_type: type[PluginSpecT],
) -> Callable[[Iterable[PluginDict]], Iterator[SymbolSpec]]:
    """Convenience function to create a `SymbolResolver`.

    As `PluginSpec` is expected to be easier to handle than a dictionary this function takes over
    the task of assembling a `PluginSpec` type object from a dictionary and pass it to the given
    function `fn`.
    """

    def resolver(data: Iterable[PluginDict]) -> Iterator[SymbolSpec]:
        plugins = map(partial(build_plugin_spec, spec_type=plugin_spec_type), data)
        for module, symbols in fn(plugins):
            yield SymbolSpec(module, symbols)

    return resolver


class PluginLoader(Generic[_Tcontra]):
    """Get a set of plugins from a package, use `extract_fn` to resolve the symbols and load them into the `plugin_receiver`.

    .Args
    * `extract_fn`: A function that extracts `SymbolSpec`s from plugin dictionaries. Use this to decide, based on the plugin specs, which symbols to load from which modules.
    * `plugin_receiver`: The object that will receive the loaded symbols. _Receive_ in this context means that the loader will call `symbol.load_into(plugin_receiver)` for each of these symbols.
    That allows symbols defined in the plugin to run code in the context of the plugin receiver.
    E.g., it could register a lowering function in a `LoweringPass` if that lowering pass is given as the receiver.
    """

    def __init__(self, extract_fn: SymbolResolver, plugin_receiver: _Tcontra) -> None:
        self._extract_fn = extract_fn
        self._receiver = plugin_receiver

    def _get_plugin_dicts(self, package: str) -> Iterable[PluginDict]:
        yield from read_plugin_dicts_from_package(package)

    def load_from_package(self, package: str) -> None:
        """load all plugins defined by the `meta.toml` file found in the package."""
        plugin_dicts = self._get_plugin_dicts(package)
        symbol_specs = self._extract_fn(plugin_dicts)
        for plugin in symbol_specs:
            if not isinstance(plugin, SymbolSpec):
                plugin = SymbolSpec(*plugin)
            module = import_module(plugin.module)
            for symbol in plugin.symbols:
                loadable = cast(PluginSymbol[_Tcontra], getattr(module, symbol))
                loadable.load_into(self._receiver)


class UnexpectedFieldError(Exception):
    def __init__(self, field_names: set[str], plugin_type: type[PluginSpecT]):
        super().__init__(
            f"unexpected fields {field_names} for plugin '{plugin_type.__qualname__}'"  # type: ignore
        )
