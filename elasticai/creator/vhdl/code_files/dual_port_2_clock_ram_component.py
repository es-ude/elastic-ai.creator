from dataclasses import dataclass

from elasticai.creator.resource_utils import read_text
from elasticai.creator.vhdl.code import Code, CodeFile
from elasticai.creator.vhdl.vhdl_files import expand_template


@dataclass
class DualPort2ClockRamComponent(CodeFile):
    layer_id: str

    @property
    def name(self) -> str:
        return f"dual_port_2_clock_ram_{self.layer_id}.vhd"

    def code(self) -> Code:
        template = read_text(
            "elasticai.creator.vhdl.templates", "dual_port_2_clock_ram.tpl.vhd"
        )
        return expand_template(template=template, layer_name=self.layer_id)
