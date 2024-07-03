import torch
from torch import nn


class BaseObserver(nn.Module):
    def __init__(self, min_float_limit: float = None, max_float_limit: float = None):
        # TODO: remove min_float_limit and max_float_limit when finished QuantisedSoftmax
        super().__init__()
        self.min_float_limit = min_float_limit
        self.max_float_limit = max_float_limit

    def UpdateFloatRange(self, running_min_float: float, running_max_float: float):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, x_r: float):
        running_min_float = torch.min(x_r)
        running_max_float = torch.max(x_r)

        self.UpdateFloatRange(running_min_float, running_max_float)


class MinMaxObserver(BaseObserver):
    def __init__(self, min_float_limit: float = None, max_float_limit: float = None):
        super().__init__(min_float_limit, max_float_limit)

        self.forward_num = (
            0  # used to initialze the min and max values for the first forward
        )
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def UpdateFloatRange(self, running_min_float: float, running_max_float: float):
        if self.forward_num == 0:
            min_float = running_min_float
            max_float = running_max_float
            self.forward_num += 1
        else:
            min_float = torch.min(running_min_float, self.min_float)
            max_float = torch.max(running_max_float, self.max_float)

        if (self.min_float_limit is not None) and (min_float < self.min_float_limit):
            min_float = self.min_float_limit
        if (self.max_float_limit is not None) and (max_float > self.max_float_limit):
            max_float = self.max_float_limit

        self.min_float.copy_(min_float)
        self.max_float.copy_(max_float)


class MovingAverageMinMaxObserver(BaseObserver):
    # TODO: BUG, fixed the bug and apply it to Activations
    def __init__(
        self,
        momentum: float,
        min_float_limit: float = None,
        max_float_limit: float = None,
    ):
        super().__init__(min_float_limit, max_float_limit)

        self.momentum = momentum
        self.forward_num = 0
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def UpdateFloatRange(self, running_min_float: float, running_max_float: float):
        if self.forward_num == 0:
            min_float = running_min_float
            max_float = running_max_float
            self.forward_num += 1
        else:
            min_float = (
                1 - self.momentum
            ) * self.min_float + self.momentum * running_min_float
            max_float = (
                1 - self.momentum
            ) * self.max_float + self.momentum * running_max_float

        if (self.min_float_limit is not None) and (min_float < self.min_float_limit):
            min_float = self.min_float_limit
        if (self.max_float_limit is not None) and (max_float > self.max_float_limit):
            max_float = self.max_float_limit

        self.min_float.copy_(min_float)
        self.max_float.copy_(max_float)
