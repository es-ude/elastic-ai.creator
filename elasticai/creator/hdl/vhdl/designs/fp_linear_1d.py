from abc import abstractmethod
from typing import Optional

from elasticai.creator.hdl.vhdl.folder import Folder
from elasticai.creator.hdl.vhdl.number_representations import FixedPointConfig
from elasticai.creator.hdl.vhdl.template import Template

from ..code_generation.code_generation import calculate_address_width
from ._network_blocks import BufferedNetworkBlock


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
            name="fp_linear1d" if name is None else name,
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
        template = Template(base_name="fp_linear_1d")
        template.update_parameters(
            layer_name=self.name,
            work_library_name=self.work_library_name,
            resource_option=f'"{self.resource_option}"',
            **self._template_parameters(),
        )
        destination.new_file(f"{self.name}", template.lines())
