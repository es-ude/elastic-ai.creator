from abc import abstractmethod
from typing import Protocol

from elasticai.creator.hdl.vhdl.file import File


class Folder(Protocol):
    @abstractmethod
    def new_file(self, name: str) -> File:
        """Throws exception if file already exists"""
        ...

    @abstractmethod
    def new_folder(self, name: str) -> "Folder":
        """Idempotent"""
        ...
