import torch

from elasticai.creator.hdl.design_base.design import Design
from elasticai.creator.hdl.translatable import Savable
from elasticai.creator.nn.vhdl.identity.design import Identity as IdentityDesign


class FPIdentity(torch.nn.Identity):
    def __init__(self, num_input_features: int, total_bits: int) -> None:
        self._num_input_features = num_input_features
        self._num_input_bits = total_bits
        super().__init__()

    def translate(self) -> Savable:
        return self.translate_to_vhdl(self.__class__.__name__.lower())

    def translate_to_vhdl(self, name: str) -> Design:
        return IdentityDesign(
            name=name,
            num_input_features=self._num_input_features,
            num_input_bits=self._num_input_bits,
        )
