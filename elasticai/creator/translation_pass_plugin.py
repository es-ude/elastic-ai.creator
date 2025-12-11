from abc import abstractmethod
from collections.abc import Callable
from typing import Protocol

import elasticai.creator.function_dispatch as F
import elasticai.creator.plugin as pl


class TranslationPass[**Tin, Tout](Protocol):
    """A translation pass that can register type handlers for translating data types."""

    @F.registrar_method
    @abstractmethod
    def register(
        self, name: str | None, fn: Callable[Tin, Tout]
    ) -> Callable[Tin, Tout]:
        """Register a type handler for the given type name."""

    @F.registrar_method
    @abstractmethod
    def override(
        self, name: str | None, fn: Callable[Tin, Tout]
    ) -> Callable[Tin, Tout]:
        """Override an existing type handler for the given type name."""


@F.registrar
def type_handler[**InT, OutT](
    name: str | None, fn: Callable[InT, OutT]
) -> pl.PluginSymbolFn[TranslationPass[InT, OutT], InT, OutT]:
    def load_into(receiver: TranslationPass[InT, OutT]) -> None:
        receiver.register(name, fn)

    return pl.make_plugin_symbol(fn=fn, load_into=load_into)


@F.registrar
def override_type_handler[**InT, OutT](
    name: str | None, fn: Callable[InT, OutT]
) -> pl.PluginSymbolFn[TranslationPass[InT, OutT], InT, OutT]:
    def load_into(receiver: TranslationPass[InT, OutT]) -> None:
        receiver.override(name, fn)

    return pl.make_plugin_symbol(fn=fn, load_into=load_into)
