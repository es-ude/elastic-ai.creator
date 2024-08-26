from pathlib import Path

from elasticai.creator.file_generation.v2.template import (
    InProjectTemplate,
    save_template,
)


class ENV5ReconfigTop:
    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name="env5_reconfig_top.vhd",
            parameters={},
        )
        save_template(template, destination.with_suffix(".vhd"))
