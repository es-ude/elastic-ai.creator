from typing import Protocol, TypeVar

T = TypeVar("T")


class IrData(Protocol[T]):
    data: dict[str, dict[str, T]]


EdgeT = TypeVar("EdgeT", bound=IrData)

PNodeT = TypeVar("PNodeT", bound=IrData)
INodeT = TypeVar("INodeT", bound=IrData)
RNodeT = TypeVar("RNodeT", bound=IrData)
GNodeT = TypeVar("GNodeT", bound=IrData)

_Tcon = TypeVar("_Tcon", contravariant=True)
_TPcon = TypeVar("_TPcon", contravariant=True)


class NodeConstraintFn(Protocol[_TPcon, _Tcon]):
    def __call__(self, pattern_node: _TPcon, graph_node: _Tcon) -> bool: ...
