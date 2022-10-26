from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class QuantizedInferenceable(Protocol):
    def quantized_forward(self, *args: Any, **kwargs: Any) -> Any:
        ...
