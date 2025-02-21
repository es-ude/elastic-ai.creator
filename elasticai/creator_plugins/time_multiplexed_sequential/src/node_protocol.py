from typing import Any, Protocol


class _NodeP(Protocol):
    implementation: str
    name: str
    type: str
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    attributes: dict[str, Any]
