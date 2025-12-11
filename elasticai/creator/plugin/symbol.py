from abc import abstractmethod
from typing import Protocol


class PluginSymbol[RecT](Protocol):
    """A symbol that the `PluginLoader` can load into a receiver object.

    The receiver can be any object.
    """

    @abstractmethod
    def load_into(self, /, receiver: RecT) -> None: ...


class PluginSymbolFn[RecT, **P, ReturnT](PluginSymbol[RecT], Protocol):
    """A `PluginSymbol` that is also a function/callable."""

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> ReturnT: ...
