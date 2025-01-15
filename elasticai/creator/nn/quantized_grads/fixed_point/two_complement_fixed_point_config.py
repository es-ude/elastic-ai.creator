import dataclasses
from dataclasses import dataclass

import torch


@dataclass
class FixedPointConfigV2:
    """
    This class behaves almost like a frozen instance,
    but calculates and sets minimum_as_rational_tensor and maximum_as_rational_tensor just once in __post__init__.
    After the __post_init__ the __setattr__ and __getattr__ methods block the change of variables.
    """

    total_bits: int
    frac_bits: int
    device: torch.device = torch.get_default_device()

    def __post_init__(self):
        """
        This python API call for dataclass is executed after init.
        This way we check if the FixedPointConfigV2 is valid while keeping the config immutable.
        """
        if self.total_bits <= 0:
            raise Exception(
                f"total bits need to be > 0 for {self.__class__.__name__}. "
                f"You have set {self.total_bits=}."
            )
        if self.frac_bits + 1 > self.total_bits:
            raise Exception(
                f"total bits-1 needs to be > frac bits for {self.__class__.__name__}."
                f"You have set {self.total_bits=} and {self.frac_bits=}."
            )
        self.minimum_as_rational_tensor = torch.Tensor(
            [-(2 ** (self.total_bits - self.frac_bits - 1))]
        ).to(device=self.device)
        self.maximum_as_rational_tensor = torch.Tensor(
            [-self.minimum_as_rational_tensor - 1 / (2**self.frac_bits)]
        ).to(device=self.device)
        self.resolution_per_int = torch.Tensor([2**self.frac_bits]).to(
            device=self.device
        )

    def __setattr__(self, name, value):
        if hasattr(self, "resolution_per_int"):
            raise dataclasses.FrozenInstanceError(self.__class__.__name__, name)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if hasattr(self, "resolution_per_int"):
            raise dataclasses.FrozenInstanceError(self.__class__.__name__, name)
        else:
            super().__delattr__(name)
