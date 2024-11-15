from elasticai.creator.file_generation.savable import Path, Savable
from elasticai.creator.file_generation.template import InProjectTemplate


class MacDesign(Savable):
    def __init__(self, name: str, vector_width: int):
        self._name = name
        self._vector_width = vector_width

    def save_to(self, destination: Path) -> None:
        core_component = InProjectTemplate(
            package="elasticai.creator.vhdl.shared_designs.mac.binary",
            file_name="bin_mac.vhd",
            parameters={
                "total_width": str(self._vector_width),
                "name": self._name,
            },
        )
        destination.create_subpath("bin_mac").as_file(".vhd").write(core_component)
