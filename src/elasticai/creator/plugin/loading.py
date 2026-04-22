from collections.abc import Iterable, Iterator
from importlib import import_module as _import_module
from typing import Any


def import_symbols(module: str, names: Iterable[str]) -> Iterator[Any]:
    """import names from a module and yield the resulting objects."""
    m = _import_module(module)
    for name in set(names):
        yield getattr(m, name)
