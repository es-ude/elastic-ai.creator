import torch
from torch import Tensor

from .compression.compression import Compression
from .delta.delta import Delta


class DeltaCompression:
    def __init__(self, delta: Delta, compression: Compression) -> None:
        self._delta = delta
        self._compression = compression

    def compress(self, input: Tensor, in_place: bool = True) -> Tensor:
        assert input.dtype == torch.int32 or input.dtype == torch.int64, (
            "Input tensor must be of type int32 or int64! Type was {}".format(
                input.dtype
            )
        )
        return self._compression.compress(self._delta.delta(input, in_place), in_place)

    def inflate(self, input: Tensor, in_place: bool = True) -> Tensor:
        assert input.dtype == torch.int32 or input.dtype == torch.int64, (
            "Input tensor must be of type int32 or int64! Type was {}".format(
                input.dtype
            )
        )
        return self._delta.reconstruct(
            self._compression.inflate(input, in_place), in_place
        )
