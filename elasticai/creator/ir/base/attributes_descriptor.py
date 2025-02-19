from collections.abc import Collection, Iterator, Mapping
from typing import Protocol

from .attribute import Attribute


class _IrData(Protocol):
    _fields: dict[str, type]
    data: dict[str, Attribute]


class _ReadOnlyMappingWithHiddenFields(Mapping[str, Attribute]):
    __slots__ = ("_mapping", "_hidden_fields")

    def __init__(
        self, mapping: Mapping[str, Attribute], hidden_fields: Collection[str]
    ):
        self._mapping = mapping
        self._hidden_fields = hidden_fields

    def __contains__(self, key: object) -> bool:
        return key in self._mapping and key not in self._hidden_fields

    def __len__(self) -> int:
        return len(set(self._mapping.keys()).difference(self._hidden_fields))

    def __iter__(self) -> Iterator[str]:
        for key in self._mapping:
            if key not in self._hidden_fields:
                yield key

    def __getitem__(self, k) -> Attribute:
        if k in self._hidden_fields:
            raise KeyError(f"key not found '{k}'.")
        return self._mapping[k]

    def __or__(self, other: Mapping[str, Attribute]) -> dict[str, Attribute]:
        return dict(self) | dict(other.items())

    def __ror__(self, other: Mapping[str, Attribute]) -> dict[str, Attribute]:
        return dict(other.items()) | dict(self)


class AttributesDescriptor:
    """Read values from the `data` dict, while hiding key, value that show up
    in `_fields` of the owning instance.

    The use for this are cases where mandatory fields are treated in a special
    way and you want write/read access to all other elements from `data`, e.g., while converting an old ir data object into a new one,
    and you just want to get the missing values from `data`.

    NOTE: The returned Mapping is read only
    """

    def __get__(self, instance: _IrData, owner=type[_IrData] | None):
        return _ReadOnlyMappingWithHiddenFields(instance.data, instance._fields)
