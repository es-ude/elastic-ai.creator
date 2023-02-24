from typing import Optional

from elasticai.creator.vhdl.code_files.utils import calculate_address_width
from elasticai.creator.vhdl.designs._network_blocks import BufferedNetworkBlock
from elasticai.creator.vhdl.designs.folder import Folder
from elasticai.creator.vhdl.number_representations import FixedPointConfig
from elasticai.creator.vhdl.templates import VHDLTemplate


class FPLinear1d(BufferedNetworkBlock):
    def __init__(
        self,
        in_feature_num: int,
        out_feature_num: int,
        fixed_point_config: FixedPointConfig,
        work_library_name: str = "work",
        resource_option: str = "auto",
        name: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            x_address_width=calculate_address_width(in_feature_num),
            y_address_width=calculate_address_width(out_feature_num),
            x_width=fixed_point_config.total_bits,
            y_width=fixed_point_config.total_bits,
        )
        self.in_feature_num = in_feature_num
        self.out_feature_num = out_feature_num
        self.work_library_name = work_library_name
        self.resource_option = resource_option
        self.frac_width = fixed_point_config.frac_bits

    @property
    def data_width(self) -> int:
        return self._x_width

    @property
    def x_addr_width(self) -> int:
        return self._x_address_width

    @property
    def y_addr_width(self) -> int:
        return self._y_address_width

    def _template_parameters(self) -> dict[str, str]:
        return dict(
            (key, str(getattr(self, key)))
            for key in (
                "data_width",
                "frac_width",
                "x_addr_width",
                "y_addr_width",
                "in_feature_num",
                "out_feature_num",
            )
        )

    def save_to(self, destination: Folder):
        template = VHDLTemplate(base_name="fp_linear_1d")
        template.update_parameters(
            layer_name=self.name,
            work_library_name=self.work_library_name,
            resource_option=f'"{self.resource_option}"',
            **self._template_parameters(),
        )
        destination.new_file(f"{self.name}", template.lines())
