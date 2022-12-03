from typing import Iterable

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.templates.utils import expand_template
from elasticai.creator.vhdl.vhdl_files import (
    VHDLModule,
    VHDLBaseModule,
    VHDLBaseFile,
)


class Module:
    def __init__(self):
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