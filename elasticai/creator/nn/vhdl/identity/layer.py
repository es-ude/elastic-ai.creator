import torch

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Translatable
from elasticai.creator.nn.vhdl.identity.design import BufferedIdentity as IdentityDesign
from elasticai.creator.nn.vhdl.identity.design import BufferlessDesign


class BufferedIdentity(Translatable, torch.nn.Identity):
    def __init__(self, num_input_features: int, total_bits: int) -> None:
        self._num_input_features = num_input_features
        self._num_input_bits = total_bits
        super().__init__()

    def translate(self, name: str) -> Design:
        return IdentityDesign(
            name=name,
            num_input_features=self._num_input_features,
            num_input_bits=self._num_input_bits,
        )


class BufferlessIdentity(Translatable, torch.nn.Identity):
    def __init__(self, total_bits: int) -> None:
        self._num_input_bits = total_bits
        super().__init__()

    def translate(self, name: str) -> Design:
        return BufferlessDesign(
            name=name,
            num_input_bits=self._num_input_bits,
        )
