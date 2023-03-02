from typing import Iterable, Optional

from elasticai.creator.hdl.vhdl.file import File
from elasticai.creator.hdl.vhdl.saveable import Path


class InMemoryFile(File):
    def __init__(self, name):
        self.text = []
        self.name = name

    def write_text(self, text: Iterable[str]) -> None:
        for line in text:
            self.text.append(line)


class InMemoryPath(Path):
    def __init__(self, name: str, parent: Optional["InMemoryPath"]):
        self.name = name
        self.children: dict[str, Path | File] = dict()
        self.parent = parent

    def as_file(self, suffix: str) -> InMemoryFile:
        file = InMemoryFile(f"{self.name}{suffix}")
        if self.parent is not None:
            self.parent.children[file.name] = file
        return file

    def create_subpath(self, subpath_name: str) -> "InMemoryPath":
        subpath = InMemoryPath(subpath_name, self)
        self.children[subpath_name] = subpath
        return subpath
