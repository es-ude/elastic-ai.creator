from collections.abc import Callable

import torch

QuantType = Callable[[torch.Tensor], torch.Tensor]
