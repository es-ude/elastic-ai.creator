import collections
from collections.abc import Iterable
from itertools import filterfalse
from typing import MutableMapping, TypeVar

T = TypeVar("T")


class _HidingDict(MutableMapping[str, T]):
    """Allows to hide keys with `hidden_names` for all read operations.
    We use this to implement an attributes field for Nodes that looks like a dictionary, but hides
    all mandatory fields.
    You can still write to `HidingDict`, e.g.,

    >>> d = dict(a="a", b="b")
    >>> h = _HidingDict({"a"}, d)
    >>> "b", == tuple(h.keys())
    True
    >>> "a" in h
    False
    >>> h["a"] = "c"
    >>> h.data["a"]
    'c'
    >>> d["a"]
    'c'
    >>> "a" in d and "a" in h.data
    True
    """

    def __init__(self, hidden_names: Iterable[str], data: dict) -> None:
        self.data = data
        self._hidden_names = set(hidden_names)

    def __setitem__(self, key: str, value: T):
        self.data[key] = value

    def _is_hidden(self, name: str) -> bool:
        return name in self._hidden_names

    def __delitem__(self, key: str):
        del self.data[key]

    def __iter__(self):
        return filterfalse(self._is_hidden, iter(self.data))

    def __contains__(self, item):
        # overriding this should also make class behave correctly for getting items
        return item not in self._hidden_names and item in self.data

    def __getitem__(self, item: str) -> T:
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def get(self, key: str, default=None) -> T:
        if key in self:
            return self[key]
        return default

    def __copy__(self) -> MutableMapping[str, T]:
        inst = self.__class__.__new__(self.__class__)
        inst.__dict__.update(self.__dict__)
        # Create a copy and avoid triggering descriptors
        inst.__dict__["data"] = self.__dict__["data"].copy()
        return inst

    def copy(self) -> MutableMapping[str, T]:
        if self.__class__ is collections.UserDict:
            return _HidingDict(self._hidden_names.copy(), self.data.copy())
        import copy

        data = self.data
        try:
            self.data = {}
            c = copy.copy(self)
        finally:
            self.data = data
        c.update(self)
        return c

    def __repr__(self) -> str:
        return f"HidingDict({', '.join(self._hidden_names)}, data={self.data})"
