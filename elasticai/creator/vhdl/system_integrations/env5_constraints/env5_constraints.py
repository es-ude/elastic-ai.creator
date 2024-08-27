from pathlib import Path

from elasticai.creator.file_generation.template import InProjectTemplate, save_template


class ENV5Constraints:
    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name="env5_constraints.xdc",
            parameters={},
        )
        save_template(template, destination.with_suffix(".xdc"))
