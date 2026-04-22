from elasticai.creator.base_modules.relu import ReLU as ReLUBase
from elasticai.creator.nn.design_creator_module import DesignCreatorModule

from .design import ReLU as ReLUDesign


class ReLU(DesignCreatorModule, ReLUBase):
    def __init__(self, total_bits: int, use_clock: bool = False) -> None:
        """Quantized Activation Function for Sigmoid
        :param total_bits:          Total number of bits
        :param use_clock:           If true, clock in the hardware design is used for shifting the output
        """
        super().__init__()
        self._total_bits = total_bits
        self._use_clock = use_clock

    def create_design(self, name: str) -> ReLUDesign:
        return ReLUDesign(
            name=name,
            total_bits=self._total_bits,
            use_clock=self._use_clock,
        )
