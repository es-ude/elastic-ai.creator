from dataclasses import dataclass

from elasticai.creator.vhdl.number_representations import FixedPointConfig, parameters
from elasticai.creator.vhdl.templates import VHDLTemplate


@dataclass
class FPReLUComponent:
    layer_id: str  # used to distinguish layers in the same model
    fixed_point_factory: FixedPointConfig

    def __post_init__(self) -> None:
        self.data_width, self.frac_width = parameters(self.fixed_point_factory)

    @property
    def name(self) -> str:
        return f"fp_relu_{self.layer_id}.vhd"

    def lines(self) -> list[str]:
        template = VHDLTemplate(base_name="fp_relu")
        template.update_parameters(
            layer_name=self.layer_id,
            data_width=str(self.data_width),
            clock_option="false",
        )

        return template.lines()
