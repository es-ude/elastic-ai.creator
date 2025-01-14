from abc import abstractmethod
from typing import Protocol


class IOStream(Protocol):
    @abstractmethod
    def write(self, data: bytes | bytearray) -> None: ...

    @abstractmethod
    def read(self, num_bytes: int) -> bytes | bytearray: ...
