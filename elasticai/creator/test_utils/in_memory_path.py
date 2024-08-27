from pathlib import Path
from typing import Optional, Protocol, TypeVar

FilesType = dict[str, "FilesValue"]
FilesValue = str | FilesType

T = TypeVar("T")


class PathProtocol(Protocol):
    def __init__(self, *pathsegments: str) -> None:
        ...

    @property
    def parent(self: T) -> T:
        ...

    @property
    def name(self) -> str:
        ...

    def joinpath(self: T, *pathsegments: str) -> T:
        ...

    def with_suffix(self: T, suffix: str) -> T:
        ...

    def __truediv__(self: T, key: str) -> T:
        ...

    def __str__(self) -> str:
        ...

    def __repr__(self) -> str:
        ...


class InMemoryPath:
    def __init__(self, *pathsegments: str, fs: Optional[FilesType] = None) -> None:
        self._segments = self._flatten_pathsegments(*pathsegments)
        self._fs = dict() if fs is None else fs

    @property
    def files(self) -> FilesType:
        return self._fs

    @property
    def parent(self) -> "InMemoryPath":
        if len(self._segments) <= 1:
            raise ValueError("Path of length 1 has no parent.")
        return self._new_path(*self._segments[:-1])

    @property
    def name(self) -> str:
        if len(self._segments) == 0:
            raise ValueError("Path of length 0 has no name.")
        return self._segments[-1]

    def joinpath(self, *pathsegments: str) -> "InMemoryPath":
        return self._new_path(*self._segments, *pathsegments)

    def with_suffix(self, suffix: str) -> "InMemoryPath":
        ...

    def __truediv__(self, key: str) -> "InMemoryPath":
        return self.joinpath(key)

    def __str__(self) -> str:
        return "/".join(self._segments)

    def _flatten_pathsegments(self, *pathsegments: str) -> list[str]:
        return [seg for path in pathsegments for seg in path.split("/")]

    def _new_path(self, *pathsegments: str) -> "InMemoryPath":
        return InMemoryPath(*pathsegments, fs=self._fs)


def print_path(path: PathProtocol) -> None:
    print(path)


def main() -> None:
    p = Path("a/b/c")
    print_path(p)


if __name__ == "__main__":
    main()
