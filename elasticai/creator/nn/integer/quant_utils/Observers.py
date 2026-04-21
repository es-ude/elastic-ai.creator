import torch
from torch import nn


class BaseObserver(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_num = 0

    def update_float_range(
        self, running_min_float: torch.FloatTensor, running_max_float: torch.FloatTensor
    ):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x: torch.FloatTensor):
        running_min_float = torch.min(x)
        running_max_float = torch.max(x)

        self.update_float_range(running_min_float, running_max_float)


class GlobalMinMaxObserver(BaseObserver):
    def __init__(self):
        super().__init__()
        " support running_min_float gets smaller and running_max_float gets larger"
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def update_float_range(
        self, running_min_float: torch.FloatTensor, running_max_float: torch.FloatTensor
    ):
        if self.forward_num == 0:
            self.min_float.copy_(running_min_float)
            self.max_float.copy_(running_max_float)
            self.forward_num += 1
        else:
            self.min_float.copy_(torch.min(running_min_float, self.min_float))
            self.max_float.copy_(torch.max(running_max_float, self.max_float))


class LocalMinMaxObserver(BaseObserver):
    def __init__(self):
        super().__init__()
        " depends on batch size"
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def update_float_range(
        self, running_min_float: torch.FloatTensor, running_max_float: torch.FloatTensor
    ):
        self.min_float.copy_(running_min_float)
        self.max_float.copy_(running_max_float)


class MovingAverageMinMaxObserver(BaseObserver):
    def __init__(
        self,
        momentum: float,
    ):
        super().__init__()

        self.momentum = momentum
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def update_float_range(
        self, running_min_float: torch.FloatTensor, running_max_float: torch.FloatTensor
    ):
        if self.forward_num == 0:
            self.min_float.copy_(running_min_float)
            self.max_float.copy_(running_max_float)
            self.forward_num += 1
        else:
            self.min_float.copy_(
                (1 - self.momentum) * self.min_float + self.momentum * running_min_float
            )
            self.max_float.copy_(
                (1 - self.momentum) * self.max_float + self.momentum * running_max_float
            )
