from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Self, cast, overload

import torch

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

    def update_path(self, path: tuple[str, ...], value: Attribute) -> Self:
        """update the entry found when following the path into nested Mappings"""

        def path_to_dict(path):
            if len(path) == 1:
                return {path[0]: value}
            return {path[0]: path_to_dict(path[1:])}

        return self.merge(path_to_dict(path))

    def merge(self, other: Mapping) -> Self:
        """merge over nested mappings recursively

        So instead of replacing a value found under a key,
        this checks wether that value is again an AttributeMapping
        and if so, updates it by the corresponding Mapping found
        in other. This happens recursively.

        Opposed to that `new_with` and the `|` operator
        only allow to update values in the most outer
        AttributeMapping. Therefore using these to update a value
        in a nested structure, users would have to recreate
        the whole outer mapping structure until they
        reach the mapping they want to update.

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
        the `update_path()` function instead to avoid
        having to build all the nested dictionaries.
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
        """replace key, value pairs in self by key, value pairs in kwargs"""
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


type AttributeConvertable = (
    Attribute | Iterable[AttributeConvertable] | Mapping[str, AttributeConvertable]
)


def is_attribute(obj: object) -> bool:
    if isinstance(obj, AttributeMapping):
        return True
    if isinstance(obj, tuple):
        if len(obj) == 0:
            return True
        else:
            return is_attribute(obj[0])
    if isinstance(obj, float | int | str | bool):
        return True
    return False


@overload
def attribute(**kwargs: AttributeConvertable) -> AttributeMapping: ...


@overload
def attribute(
    mapping: Mapping[str, AttributeConvertable], /, **kwargs: AttributeConvertable
) -> AttributeMapping: ...


@overload
def attribute(attribute: Attribute, /) -> Attribute: ...


def attribute(
    arg: Mapping[str, AttributeConvertable]
    | AttributeMapping
    | Attribute
    | None = None,
    /,
    **kwargs: AttributeConvertable,
) -> AttributeMapping | tuple[Attribute] | Attribute:
    """Create AttributeMapping from other (native) data types recursively.

    The implementation assumes that any encountered AttributeMapping objects
    are correct, ie. they only contain Attributes themselves.
    """

    def convert(arg):
        if not isinstance(arg, AttributeMapping):
            if isinstance(arg, Mapping):
                return AttributeMapping(**{k: attribute(arg[k]) for k in arg})
            elif (
                isinstance(arg, Iterable)
                and not isinstance(arg, str)
                and not isinstance(arg, torch.Tensor)
            ):
                return tuple(convert(item) for item in arg)  # type: ignore
            elif isinstance(arg, torch.Tensor):
                return arg.detach().numpy().tolist()
        return arg

    if isinstance(arg, Mapping):
        if len(kwargs) > 0:
            arg = kwargs | dict(arg)
        return convert(arg)
    elif arg is None:
        return convert(kwargs)
    else:
        if len(kwargs) > 0:
            raise TypeError(
                "unsupported call of attribute, only use kwargs if you are creating a mapping"
            )
        return convert(arg)
