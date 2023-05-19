from typing import Optional, Protocol

import torch


class Arithmetics(Protocol):
    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        ...

    def clamp(self, a: torch.Tensor) -> torch.Tensor:
        ...

    def round(self, a: torch.Tensor) -> torch.Tensor:
        ...

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...

    def sum(
        self, a: torch.Tensor, dim: Optional[int | tuple[int, ...]]
    ) -> torch.Tensor:
        ...

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ...
