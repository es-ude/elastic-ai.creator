from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)


class LSTMSkeleton:
    def __init__(self, network_name: str):
        self.name = "skeleton"
        self._network_name = network_name

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstm_network_skeleton.tpl.vhd",
            parameters=dict(name=self.name, network_name=self._network_name),
        )
        file = destination.as_file(".vhd")
        file.write(template)
