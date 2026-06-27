from abc import ABC, abstractmethod

from torch import Tensor


class Delta(ABC):
    @abstractmethod
    def delta(self, input: Tensor, in_place: bool) -> Tensor: ...

    @abstractmethod
    def reconstruct(self, input: Tensor, in_place: bool) -> Tensor: ...
