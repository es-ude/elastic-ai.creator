from pathlib import Path

from elasticai.creator.file_generation.v2.template import (
    InProjectTemplate,
    save_template,
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
                package=InProjectTemplate.module_to_package(self.__module__),
                file_name=name + ".vhd",
                parameters={},
            )
            save_template(template, destination / f"{name}.vhd")
