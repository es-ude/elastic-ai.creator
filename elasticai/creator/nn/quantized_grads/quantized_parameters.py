from typing import Callable

import torch


class QuantizationSchemeByName:
    __slots__ = ("name", "quantization")

    def __init__(self, name: str, quantization: Callable[[torch.Tensor], None]):
        self.name = name
        self.quantization = quantization


class QuantizationParameter:
    __slots__ = ("name", "quantization", "parameter", "gradient")

    def __init__(
        self,
        name: str,
        quantization: Callable[[torch.Tensor], None],
        parameter: torch.Tensor,
        gradient: torch.Tensor,
    ):
        self.name = name
        self.quantization = quantization
        self.parameter = parameter
        self.gradient = gradient


class QuantizedParameters:
    __slots__ = "qparams"

    def __init__(self, qparams: list[QuantizationSchemeByName], *args, **kwargs):
        self.qparams: list[QuantizationSchemeByName] = qparams
        print("INIT DONE")
