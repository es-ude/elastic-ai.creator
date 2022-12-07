from itertools import chain
from typing import Iterable

import torch.nn

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.components.network_component import (
    SignalsForComponentWithBuffer,
    SignalsForBufferlessComponent,
    BufferedComponentInstantiation,
    ComponentInstantiation,
)
from elasticai.creator.vhdl.modules.hard_sigmoid import FixedPointHardSigmoid
from elasticai.creator.vhdl.modules.linear import FixedPointLinear
from elasticai.creator.vhdl.modules.relu import FixedPointReLU
from elasticai.creator.vhdl.templates.utils import (
    expand_template,
    expand_multiline_template,
)
from elasticai.creator.vhdl.vhdl_files import VHDLModule, VHDLBaseModule, VHDLBaseFile


class RootModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.elasticai_tags = {
            "x_address_width": 1,
            "y_address_width": 1,
            "data_width": 1,
        }

    def to_vhdl(self) -> Iterable[VHDLModule]:
        code = read_text("elasticai.creator.vhdl.templates", "network.tpl.vhd")
        code = expand_template(code, **self.elasticai_tags)
        signals = SignalsForComponentWithBuffer(
            name="fp_linear",
            x_address_width=self.elasticai_tags["x_address_width"],
            y_address_width=self.elasticai_tags["y_address_width"],
            data_width=self.elasticai_tags["data_width"],
        )
        sigmoid_signals = SignalsForBufferlessComponent(
            name="fp_hard_sigmoid", data_width=self.elasticai_tags["data_width"]
        )
        signals = chain(signals.code(), sigmoid_signals.code())
        layer_instantiations = chain(
            BufferedComponentInstantiation("fp_linear").code(),
            ComponentInstantiation("fp_hard_sigmoid").code(),
        )
        code = expand_multiline_template(
            code, signal_definitions=signals, layer_instantiations=layer_instantiations
        )

        yield from [
            VHDLBaseModule(
                name="network",
                files=(VHDLBaseFile(name="network.vhd", code=code),),
            )
        ]
