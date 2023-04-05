import os
from pathlib import Path as _PyPath
from typing import Iterable

from elasticai.creator.hdl.translatable import File, Path


class OnDiskFile(File):
    def __init__(self, full_path: str):
        self._full_path = full_path

    def write_text(self, text: Iterable[str]) -> None:
        full_path = _PyPath(self._full_path)
        folder = full_path.parent
        if not folder.exists():
            os.makedirs(folder)
        with open(full_path, "w") as f:
            f.writelines((f"{line}\n" for line in text))


class OnDiskPath(Path):
    def __init__(self, name: str, parent: str = "."):
        self._full_path = f"{parent}/{name}"

    def create_subpath(self, name: str) -> "Path":
        return OnDiskPath(name, parent=self._full_path)

    def as_file(self, suffix: str) -> "File":
        return OnDiskFile(full_path=f"{self._full_path}{suffix}")
