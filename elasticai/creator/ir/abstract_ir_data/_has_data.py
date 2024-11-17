from typing import Protocol, runtime_checkable

from elasticai.creator.ir.attribute import AttributeT


@runtime_checkable
class HasData(Protocol):
    @property
    def data(self) -> dict[str, AttributeT]: ...
