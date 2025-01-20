from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from elasticai.creator.function_utils import FunctionDecoratorFactory
from elasticai.creator.ir import Lowerable
from elasticai.creator.ir import LoweringPass as _LoweringPass
from elasticai.creator.plugin import PluginLoader as _PluginLoader
from elasticai.creator.plugin import PluginSpec as _BasePluginSpec
from elasticai.creator.plugin import PluginSymbol, make_symbol_resolver

Tin = TypeVar("Tin", bound=Lowerable)
Tout = TypeVar("Tout")


@dataclass
class PluginSpec(_BasePluginSpec):
    generated: tuple[str, ...]
    templates: tuple[str, ...]
    static_files: tuple[str, ...]


class PluginLoader(_PluginLoader, Generic[Tin, Tout]):
    """Connects a LoweringPass and corresponding plugins.

    Plugins can provide types that will be registered at the lowering pass.
    The lowering pass will subsequently use the registered functions
    to lower `Tin` to `Tout`.
    """

    def __init__(self, target_runtime: str, lowering: _LoweringPass[Tin, Tout]):
        super().__init__(
            extract_fn=make_symbol_resolver(self._extract_symbols, PluginSpec),
            plugin_receiver=self,
        )
        self._lowering = lowering
        self.target_runtime = target_runtime

    def _extract_symbols(
        self, data: Iterable[PluginSpec]
    ) -> Iterable[tuple[str, set[str]]]:
        for p in data:
            if p.target_runtime == self.target_runtime:
                module = f"{p.package}.src"
                yield module, set(p.generated)

    def register_iterable(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._lowering.register_iterable(name)(fn)

    def register(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._lowering.register(name)(fn)


class _GeneratedCodeType(PluginSymbol, Generic[Tin, Tout]):
    """
    Acts as a thin wrapper around a given function
    `fn`. It is used to provide an easy to use
    default `load` mechanism for plugins that
    want to provide lowering functions to `LoweringPass`es.
    """

    def __init__(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Tout:
        return self._fn(arg)

    def load_into(self, lower: PluginLoader[Tin, Tout]) -> None:
        lower.register(self._name, self._fn)


class _GeneratedIterableCodeType(PluginSymbol, Generic[Tin, Tout]):
    def __init__(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Iterable[Tout]:
        return self._fn(arg)

    def load(self, lower: PluginLoader[Tin, Tout]) -> None:
        lower.register_iterable(self._name, self._fn)


class TypeHandlerDecorator(
    FunctionDecoratorFactory[Callable[[Tin], Tout], _GeneratedCodeType[Tin, Tout]],
    Generic[Tin, Tout],
):
    """Mark a function as a type handler for a lowering pass."""

    def __init__(self):
        super().__init__(_GeneratedCodeType)


class IterableTypeHandlerDecorator(
    FunctionDecoratorFactory[
        Callable[[Tin], Iterable[Tout]], _GeneratedIterableCodeType[Tin, Tout]
    ],
    Generic[Tin, Tout],
):
    """Mark a function that returns an Iterable[Tout] as a type handler for a lowering pass."""

    def __init__(self):
        super().__init__(_GeneratedCodeType)


type_handler: TypeHandlerDecorator = TypeHandlerDecorator()

iterable_type_handler: IterableTypeHandlerDecorator = IterableTypeHandlerDecorator()
