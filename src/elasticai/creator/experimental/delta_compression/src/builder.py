from typing import Self

from .compression.bitmask import BitMaskCompression
from .compression.compression import Compression
from .compression.saturate import SaturatedCompression
from .delta.consecutive_delta import ConsecutiveDelta
from .delta.delta import Delta
from .delta.fixed_reference_delta import FixedReferenceDelta
from .delta_compression import DeltaCompression


class DeltaCompBuilder:
    def __init__(self) -> None:
        self._delta: Delta | None = None
        self._compression: Compression | None = None

    def consecutive_delta(self) -> Self:
        if self._delta is not None:
            raise RuntimeError("Delta method already set!")
        self._delta = ConsecutiveDelta()
        return self

    def fixed_reference_delta(self) -> Self:
        if self._delta is not None:
            raise RuntimeError("Delta method already set!")
        self._delta = FixedReferenceDelta()
        return self

    def saturated_compression(self, delta_width: int, offset: int) -> Self:
        if self._compression is not None:
            raise RuntimeError("Compression method already set!")
        if delta_width <= 0:
            raise ValueError("delta_width must be greater to 0!")
        if offset < 0:
            raise ValueError("offset mus be greater of equal to 0!")

        self._compression = SaturatedCompression(delta_width, offset)
        return self

    def bitmask_compression(self, delta_width: int, offset: int) -> Self:
        if self._compression is not None:
            raise RuntimeError("Compression method already set!")
        if delta_width <= 0:
            raise ValueError("delta_width must be greater to 0!")
        if offset < 0:
            raise ValueError("offset mus be greater of equal to 0!")

        self._compression = BitMaskCompression(delta_width, offset)
        return self

    def build(self) -> DeltaCompression:
        if self._delta is None:
            raise RuntimeError("Missing Delta Method!")
        if self._compression is None:
            raise RuntimeError("Missing Compression Method!")

        return DeltaCompression(
            self._delta,
            self._compression,
        )
