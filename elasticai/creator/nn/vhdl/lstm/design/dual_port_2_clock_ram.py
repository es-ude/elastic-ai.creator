from elasticai.creator.hdl.code_generation.abstract_base_template import (
    TemplateConfig,
    TemplateExpander,
    module_to_package,
)
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.translatable import Path


class DualPort2ClockRam(Design):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def port(self) -> Port:
        return Port(incoming=[], outgoing=[])

    def save_to(self, destination: Path) -> None:
        template_configuration = TemplateConfig(
            file_name="dual_port_2_clock_ram.tpl.vhd",
            package=module_to_package(self.__module__),
            parameters=dict(name=self.name),
        )
        expander = TemplateExpander(template_configuration)
        destination.as_file(".vhd").write_text(expander.lines())
