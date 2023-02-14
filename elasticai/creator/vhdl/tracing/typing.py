import typing
from abc import abstractmethod
from typing import Iterable, Optional, Protocol, overload, runtime_checkable

from elasticai.creator.mlframework import Module
from elasticai.creator.vhdl.code import Translatable


@runtime_checkable
class TranslatableLayer(Translatable, Module, Protocol):
    ...


T_Module = typing.TypeVar("T_Module", bound=Module, covariant=True)


@runtime_checkable
class Node(Protocol):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def op(self) -> str:
        ...


class HWEquivalentTracer(Protocol[T_Module]):
    @abstractmethod
    def trace(self, root: Module, **kwargs) -> "HWEquivalentGraph[T_Module]":
        ...


class HWEquivalentGraph(Protocol[T_Module]):
    @overload
    def get_module_for_node(self, node: str) -> Optional[T_Module]:
        ...

    @overload
    def get_module_for_node(self, node: Node) -> Optional[T_Module]:
        ...

    @abstractmethod
    def get_module_for_node(self, node: Node | str) -> Optional[T_Module]:
        ...

    @overload
    def node_has_module(self, node: str) -> bool:
        ...

    @overload
    def node_has_module(self, node: Node) -> bool:
        ...

    @abstractmethod
    def node_has_module(self, node: str | Node) -> bool:
        ...

    @property
    @abstractmethod
    def module_nodes(self) -> Iterable[Node]:
        ...
