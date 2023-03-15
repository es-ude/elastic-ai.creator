from typing import Iterable, Optional

from elasticai.creator.hdl.translatable import File, Path


class InMemoryFile(File):
    def __init__(self, name):
        self.text: list[str] = []
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
        if len(self.children) > 0:
            raise ValueError(
                "non empty path {}, present children: {}".format(
                    self.name, ", ".join(self.children)
                )
            )
        if self.parent is not None:
            self.parent.children[self.name] = file
        return file

    def __getitem__(self, item):
        return self.children[item]

    def create_subpath(self, subpath_name: str) -> "InMemoryPath":
        subpath = InMemoryPath(name=subpath_name, parent=self)
        self.children[subpath_name] = subpath
        return subpath
