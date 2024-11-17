from collections.abc import MutableMapping

from elasticai.creator.ir.attribute import AttributeT

from ._has_data import HasData
from ._hiding_dict import _HidingDict


class _AttributesDescriptor:
    def __init__(self, hidden_names: set[str]):
        self._hidden_names = hidden_names

    def __get__(
        self, instance: HasData, owner: type[HasData]
    ) -> MutableMapping[str, AttributeT]:
        return _HidingDict(self._hidden_names, instance.data)
