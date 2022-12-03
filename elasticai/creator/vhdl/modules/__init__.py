from typing import Iterable

import torch.nn

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.modules.hard_sigmoid import FixedPointHardSigmoid
from elasticai.creator.vhdl.modules.linear import FixedPointLinear
from elasticai.creator.vhdl.modules.relu import FixedPointReLU
from elasticai.creator.vhdl.templates.utils import expand_template
from elasticai.creator.vhdl.vhdl_files import VHDLModule, VHDLBaseModule, VHDLBaseFile


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.elasticai_tags = {}

    def to_vhdl(self) -> Iterable[VHDLModule]:
        code = read_text("elasticai.creator.vhdl.templates", "network.tpl.vhd")
        code = expand_template(code, **self.elasticai_tags)

        yield from [
            VHDLBaseModule(
                name="network",
                files=(VHDLBaseFile(name="network.vhd", code=code),),
            )
        ]
