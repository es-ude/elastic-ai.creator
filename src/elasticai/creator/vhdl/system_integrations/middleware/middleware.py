from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)


class Middleware:
    def save_to(self, destination: Path):
        file_names = [
            "icapInterface",
            "InterfaceStateMachine",
            "middleware",
            "spi_slave",
            "UserLogicInterface",
        ]
        for name in file_names:
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name=name + ".vhd",
                parameters={},
            )
            file = destination.create_subpath(name).as_file(".vhd")
            file.write(template)
