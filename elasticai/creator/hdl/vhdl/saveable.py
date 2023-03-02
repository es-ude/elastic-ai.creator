from abc import ABC, abstractmethod
from typing import Protocol

from elasticai.creator.hdl.vhdl.file import File
from elasticai.creator.hdl.vhdl.folder import Folder


class Path(Protocol):
    @abstractmethod
    def as_file(self, suffix: str) -> File:
        ...

    @abstractmethod
    def create_subpath(self, subpath_name: str) -> "Path":
        ...


class Saveable(ABC):
    @abstractmethod
    def save_to(self, destination: Path):
        ...
