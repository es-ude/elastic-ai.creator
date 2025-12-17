from collections.abc import Iterator, Mapping
from typing import Any, Self, cast

type AttributeBaseData = float | int | str | bool

type AttributeTuple = tuple["Attribute", ...]

type Attribute = AttributeBaseData | "AttributeTuple" | "AttributeMapping"


class AttributeMapping(Mapping[str, Attribute]):
    def __init__(self, **kwargs: Attribute) -> None:
        self._mapping = dict(kwargs)

    def __getitem__(self, key: str) -> Any:
        # There is no feasible way to type this properly.
        # As values are highly dynamic. TypedDict would not
        # work here because they are mutable and would
        # require all keys to be known at runtime adding
        # unreasonable overhead for users. Alternatively,
        # we could return Attribute, but that would will
        # result in many false positives in type checking.
        # Users would have to cast the result almost everytime.
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping)

    def __len__(self) -> int:
        return len(self._mapping)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, AttributeMapping):
            return self._mapping == other._mapping
        if isinstance(other, Mapping):
            return self._mapping == dict(other)
        return False

    def __repr__(self) -> str:
        return f"AttributeMapping({repr(self._mapping)})"

    def drop(self, key: str) -> Self:
        new_dict = {k: v for k, v in self._mapping.items() if k != key}
        return type(self)(**new_dict)

    def __or__(self, other: object) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        if len(other) == 0:
            return self
        return type(self)(
            **(self._mapping | dict(cast(Mapping[str, Attribute], other)))
        )

    def new_with(self, **kwargs: Attribute) -> "AttributeMapping":
        return self | kwargs
