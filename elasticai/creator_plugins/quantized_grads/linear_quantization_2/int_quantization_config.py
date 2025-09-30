from dataclasses import dataclass

from torch import Tensor

from elasticai.creator_plugins.quantized_grads.linear_quantization_2.linear_quantization_config import \
    LinearQuantizationConfig


@dataclass(frozen=True)
class IntQuantizationConfig(LinearQuantizationConfig):
    @property
    def min_value(self) -> Tensor:
        return Tensor([-2**(self.num_bits-1)])

    @property
    def max_value(self)-> Tensor:
        return Tensor([2**(self.num_bits-1)-1])
