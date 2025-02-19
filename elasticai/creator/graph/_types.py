from typing import Protocol, TypeVar

T = TypeVar("T")


class IrData(Protocol[T]):
    data: dict[str, dict[str, T]]


EdgeT = TypeVar("EdgeT", bound=IrData)

PNodeT = TypeVar("PNodeT", bound=IrData)
INodeT = TypeVar("INodeT", bound=IrData)
RNodeT = TypeVar("RNodeT", bound=IrData)
GNodeT = TypeVar("GNodeT", bound=IrData)
