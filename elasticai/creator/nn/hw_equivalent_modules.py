from elasticai.creator.vhdl.modules.linear import FixedPointLinear as _FixedPointLinear
from vhdl.hw_equivalent_layers.hw_blocks import (
    _SignalsForComponentWithBuffer,
    _BufferedComponentInstantiation,
)
from vhdl.code_files.utils import calculate_address_width
from vhdl.code import Code


class FixedPointLinear(_FixedPointLinear):
    def signals(self, prefix) -> Code:
        signals = _SignalsForComponentWithBuffer(
            name=prefix,
            data_width=self.fixed_point_factory.total_bits,
            x_address_width=calculate_address_width(self.in_features),
            y_address_width=calculate_address_width(self.out_features),
        )
        yield from signals

    def instantiation(self, prefix) -> Code:
        instantiation = _BufferedComponentInstantiation(
            name=prefix,
        )
        yield from instantiation
