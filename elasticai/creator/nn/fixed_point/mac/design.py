from elasticai.creator.file_generation.savable import Path, Savable
from elasticai.creator.file_generation.template import InProjectTemplate


class MacDesign(Savable):
    def __init__(self, total_bits, frac_bits):
        pass

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="fxp_mac.tpl.vhd",
            parameters={},
        )
        destination.create_subpath("fxp_mac").as_file(".vhd").write(template)
