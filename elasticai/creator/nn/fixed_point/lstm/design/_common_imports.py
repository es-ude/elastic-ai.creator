from functools import partial

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.fixed_point._two_complement_fixed_point_config import (
    FixedPointConfig,
)
from elasticai.creator.nn.fixed_point.hard_sigmoid import HardSigmoid
from elasticai.creator.nn.fixed_point.hard_tanh.design import HardTanh
from elasticai.creator.nn.fixed_point.linear.design import LinearDesign as FPLinear1d
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design import std_signals
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port
from elasticai.creator.vhdl.design.signal import Signal
from elasticai.creator.vhdl.shared_designs.rom import Rom
