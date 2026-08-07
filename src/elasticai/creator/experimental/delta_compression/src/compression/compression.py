from abc import ABC, abstractmethod

from torch import Tensor


class Compression(ABC):
    @abstractmethod
    def compress(self, input: Tensor, in_place: bool = True) -> Tensor: ...

    @abstractmethod
    def inflate(self, input: Tensor, in_place: bool = True) -> Tensor: ...
