from pathlib import Path

from elasticai.creator.file_generation.v2.savable import Savable
from elasticai.creator.file_generation.v2.template import (
    InProjectTemplate,
    save_template,
)


class MacDesign(Savable):
    def __init__(self, name: str, vector_width: int):
        self._name = name
        self._vector_width = vector_width

    def save_to(self, destination: Path) -> None:
        core_component = InProjectTemplate(
            package="elasticai.creator.nn.binary.mac",
            file_name="bin_mac.vhd",
            parameters={
                "total_width": str(self._vector_width),
                "name": self._name,
            },
        )
        save_template(core_component, destination / "bin_mac.vhd")
