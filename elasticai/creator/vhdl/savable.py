from abc import abstractmethod
from typing import Protocol

from elasticai.creator.vhdl.code_generation.template import Template


class File(Protocol):
    @abstractmethod
    def write(self, template: Template) -> None:
        ...


class Path(Protocol):
    @abstractmethod
    def as_file(self, suffix: str) -> File:
        ...

    @abstractmethod
    def create_subpath(self, subpath_name: str) -> "Path":
        ...


class Savable(Protocol):
    @abstractmethod
    def save_to(self, destination: Path) -> None:
        ...
