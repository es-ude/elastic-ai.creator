from collections.abc import Iterator, Mapping
from typing import Any, Self, cast, overload

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

    def get(self, key: str, default: Attribute) -> Any:
        if key in self._mapping:
            return self._mapping
        else:
            return default

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

    def update_path(self, path: tuple[str, ...], value: Attribute) -> Self:
        """update the entry found when following the path into nested Mappings"""

        def path_to_dict(path):
            if len(path) == 0:
                return value
            return {path[0]: path_to_dict(path[1:])}

        return self.merge(path_to_dict(path))

    def merge(self, other: Mapping) -> Self:
        """only update values that are present in other

        Use Case:

        If you would want to write

        ```python
        data = dict(a=dict(b=1, c=2))
        data["a"]["b"] = 3

        assert data["a"]["c"] == 2
        ```

        You can instead

        ```python
        data = AttributeMapping(a=AttributeMapping(b=1, c=2))
        data = data.merge(dict(a=dict(b=3)))
        assert data["a"]["c"] == 2
        ```

        If you want to update a single value, you can use
        the `update_path()` function instead.
        """
        mapping = dict((k, v) for k, v in self._mapping.items())
        for k in mapping:
            if k in other:
                item = mapping[k]
                if isinstance(item, AttributeMapping):
                    mapping[k] = item.merge(other[k])
                else:
                    mapping[k] = other[k]
        return type(self)(**mapping)

    def new_with(self, **kwargs: Attribute) -> "AttributeMapping":
        return self | kwargs

    @classmethod
    def from_dict(cls, data: dict) -> "AttributeMapping":
        @overload
        def to_attribute(data: dict) -> AttributeMapping: ...

        @overload
        def to_attribute(data: Attribute | list) -> Attribute: ...

        def to_attribute(data: Attribute | dict | list) -> Attribute:
            if isinstance(data, dict):
                return cls(**{k: to_attribute(v) for k, v in data.items()})
            elif isinstance(data, list):
                return tuple(to_attribute(v) for v in data)
            else:
                return data

        return to_attribute(data)
