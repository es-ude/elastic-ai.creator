from elasticai.creator.file_generation.savable import Path, Savable
from elasticai.creator.file_generation.template import InProjectTemplate


class MacDesign(Savable):
    def __init__(self, name: str, vector_width: int, fxp_params):
        self._name = name
        self._vector_width = vector_width
        self._fxp_params = fxp_params

    def save_to(self, destination: Path) -> None:
        wrapper = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="mac_design.tpl.vhd",
            parameters={
                "total_width": str(self._fxp_params.total_bits),
                "frac_width": str(self._fxp_params.frac_bits),
                "vector_width": str(self._vector_width),
                "name": self._name,
            },
        )
        core_component = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="fxp_mac.tpl.vhd",
            parameters={},
        )
        destination.create_subpath("fxp_mac").as_file(".vhd").write(core_component)
        destination.create_subpath(self._name).as_file(".vhd").write(wrapper)
