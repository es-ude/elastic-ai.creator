from creator.file_generation.savable import Path, Savable
from creator.file_generation.template import InProjectTemplate
from fixed_point.mac._signal_number_converter import SignalNumberConverter


class MacDesign(Savable):
    def __init__(self, total_bits, frac_bits):
        pass

    @staticmethod
    def signal_converter():
        return SignalNumberConverter(total_bits=4, frac_bits=2)

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package="elasticai.creator.nn.fixed_point.mac",
            file_name="fxp_mac.tpl.vhd",
            parameters={},
        )
        destination.create_subpath("fxp_mac").as_file(".vhd").write(template)
