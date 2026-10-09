from abc import ABC, abstractmethod

import torch
from torch import nn

from elasticai.creator.nn.integer._dtypes import FLOAT_DTYPES, check_dtype


class BaseObserver(nn.Module, ABC):
    """Base class for observers that track the float range of a tensor.

    Subclasses store the range in the `min_float` and `max_float` buffers and
    implement `update_float_range`.
    """

    min_float: torch.Tensor
    max_float: torch.Tensor
    initialized: torch.Tensor

    def __init__(self):
        super().__init__()
        # A buffer, not a plain attribute, so that it is saved in the state dict
        # and a loaded range is extended instead of overwritten.
        self.register_buffer("initialized", torch.tensor(False))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints saved before `initialized` existed hold an observed range.
        state_dict.setdefault(prefix + "initialized", torch.tensor(True))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @abstractmethod
    def update_float_range(
        self, running_min_float: torch.Tensor, running_max_float: torch.Tensor
    ) -> None:
        """Update the stored float range with the range of the current batch.

        Args:
            running_min_float: Minimum of the current batch.
            running_max_float: Maximum of the current batch.

        Raises:
            NotImplementedError: Always; subclasses must override this method.
        """
        ...

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> None:
        """Observe `x` and update the stored float range.

        Args:
            x: Float tensor whose minimum and maximum are observed.

        Raises:
            TypeError: If `x` is not a `torch.Tensor` or does not have a
                floating-point dtype.
        """
        check_dtype("x", x, *FLOAT_DTYPES)
        running_min_float = torch.min(x)
        running_max_float = torch.max(x)

        self.update_float_range(running_min_float, running_max_float)


class GlobalMinMaxObserver(BaseObserver):
    """Observer that keeps the smallest minimum and largest maximum ever seen."""

    def __init__(self):
        super().__init__()
        # supports running_min_float getting smaller and running_max_float larger
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def update_float_range(
        self, running_min_float: torch.Tensor, running_max_float: torch.Tensor
    ) -> None:
        """Widen the stored range to include the current batch.

        The first call stores the batch range as is.

        Args:
            running_min_float: Minimum of the current batch.
            running_max_float: Maximum of the current batch.
        """
        if not self.initialized:
            self.min_float.copy_(running_min_float)
            self.max_float.copy_(running_max_float)
            self.initialized.fill_(True)
        else:
            self.min_float.copy_(torch.min(running_min_float, self.min_float))
            self.max_float.copy_(torch.max(running_max_float, self.max_float))


class LocalMinMaxObserver(BaseObserver):
    """Observer that keeps only the range of the most recent batch."""

    def __init__(self):
        super().__init__()
        # depends on batch size
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def update_float_range(
        self, running_min_float: torch.Tensor, running_max_float: torch.Tensor
    ) -> None:
        """Replace the stored range with the range of the current batch.

        Args:
            running_min_float: Minimum of the current batch.
            running_max_float: Maximum of the current batch.
        """
        self.min_float.copy_(running_min_float)
        self.max_float.copy_(running_max_float)


class MovingAverageMinMaxObserver(BaseObserver):
    """Observer that tracks an exponential moving average of the range.

    Args:
        momentum: Weight of the current batch in the moving average.
    """

    def __init__(
        self,
        momentum: float,
    ):
        super().__init__()

        self.momentum = momentum
        self.register_buffer("min_float", torch.zeros((1), dtype=torch.float32))
        self.register_buffer("max_float", torch.zeros((1), dtype=torch.float32))

    def update_float_range(
        self, running_min_float: torch.Tensor, running_max_float: torch.Tensor
    ) -> None:
        """Blend the current batch range into the stored range.

        The first call stores the batch range as is. Later calls compute
        `(1 - momentum) * stored + momentum * current` for both bounds.

        Args:
            running_min_float: Minimum of the current batch.
            running_max_float: Maximum of the current batch.
        """
        if not self.initialized:
            self.min_float.copy_(running_min_float)
            self.max_float.copy_(running_max_float)
            self.initialized.fill_(True)
        else:
            self.min_float.copy_(
                (1 - self.momentum) * self.min_float + self.momentum * running_min_float
            )
            self.max_float.copy_(
                (1 - self.momentum) * self.max_float + self.momentum * running_max_float
            )
