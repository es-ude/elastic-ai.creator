from typing import Any, Callable, Protocol

import torch

QuantType = Callable[[torch.Tensor], torch.Tensor]


class QuantizedModule(Protocol):
    def quantized_forward(self, *args: Any, **kwargs: Any) -> Any:
        ...
