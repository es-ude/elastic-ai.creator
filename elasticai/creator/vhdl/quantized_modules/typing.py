from typing import Callable

import torch

QuantType = Callable[[torch.Tensor], torch.Tensor]
OperationType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
