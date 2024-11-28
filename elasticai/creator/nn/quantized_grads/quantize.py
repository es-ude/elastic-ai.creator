from typing import Protocol

from elasticai.creator.nn.quantized_grads.quantization_config import QuantizationConfig


class Quantize(Protocol):
    config: QuantizationConfig

    def quantize(self): ...

    def clamp(self): ...
