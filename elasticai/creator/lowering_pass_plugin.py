from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

from elasticai.creator.ir import Lowerable
from elasticai.creator.ir import LoweringPass as _LoweringPass
from elasticai.creator.function_utils import FunctionDecoratorFactory
from elasticai.creator.plugin import (
    BasePluginLoader as _BasePluginLoader,
)
from elasticai.creator.plugin import (
    Plugin as _BasePlugin,
)

Tin = TypeVar("Tin", bound=Lowerable)
Tout = TypeVar("Tout")


@dataclass(frozen=True)
class SubFolderStructure:
    generated: str
    templates: str
    static_files: str


@dataclass
class Plugin(_BasePlugin):
    generated: list[str]
    templates: list[str]
    static_files: list[str]


class Loader(_BasePluginLoader["Loader", Plugin], ABC, Generic[Tin, Tout]):
    """Connects a LoweringPass and corresponding plugins.

    Plugins can provide types that will be registered at the lowering pass.
    The lowering pass will subsequently use the registered functions
    to lower `Tin` to `Tout`.
    """

    def __init__(self, lowering: _LoweringPass[Tin, Tout]):
        super().__init__(plugin_type=Plugin)
        self._lowering = lowering

    @property
    @abstractmethod
    def folders(self) -> SubFolderStructure: ...

    def _get_loadables(self, p: Plugin) -> dict[str, set[str]]:
        module = f"{p.package}.{self.folders.generated}"
        loadables: dict[str, set[str]] = {module: set()}
        for name in p.generated:
            loadables[module].add(name)
        return loadables

    def register_iterable(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._lowering.register_iterable(name)(fn)

    def register(self, name: str, fn: Callable[[Tin], Tout]) -> None:
        self._lowering.register(name)(fn)


class _GeneratedCodeType(Generic[Tin, Tout]):
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

    def load(self, lower: Loader[Tin, Tout]) -> None:
        lower.register(self._name, self._fn)


class _GeneratedIterableCodeType(Generic[Tin, Tout]):
    def __init__(self, name: str, fn: Callable[[Tin], Iterable[Tout]]) -> None:
        self._fn = fn
        self._name = name

    def __call__(self, arg: Tin, /) -> Iterable[Tout]:
        return self._fn(arg)

    def load(self, lower: Loader[Tin, Tout]) -> None:
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
