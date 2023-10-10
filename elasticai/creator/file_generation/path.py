import pathlib
from typing import ContextManager, Protocol

from creator.file_generation.template import Template


class TextIO(ContextManager, Protocol):
    def close(self):
        ...

    def write(self, text: str):
        ...


class TemplateIO(Protocol):
    def write(self, text: Template | str):
        ...


class Path(Protocol):
    def create_subpath(self, name: str) -> "Path":
        """
        Similar to python's pathlib.Path.joinpath function.
        E.g., assuming an underlying path of `build_dir/network` calling `create_subpath("submodule")`
        returns an object representing `build_dir/network/my_submodule`.
        """
        ...

    def open(self, mode: str = "w") -> TextIO:
        """
        Opens the path for writing.
        This returns an object that has a `write` method taking a `str`
        and a `close` method.
        Additionally, the object implements the ContextManager interface,
        so we can use it in a `with` statement.
        """
        ...


class OnDiskPath:
    def __init__(self, path: pathlib.Path):
        self._path = path

    def create_subpath(self, name: str) -> "Path":
        return OnDiskPath(self._path.joinpath(name))

    def open(self, mode: str = "w") -> TextIO:
        return self._path.open(mode=mode)
