from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)


class ENV5Constraints:
    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="env5_constraints.xdc",
            parameters={},
        )
        file = destination.as_file(".xdc")
        file.write(template)
