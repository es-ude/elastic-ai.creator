from typing import Optional

from elasticai.creator.hdl.code_generation.abstract_base_template import (
    module_to_package,
)
from elasticai.creator.hdl.design_base.design import Design, Port
from elasticai.creator.hdl.design_base.ports import (
    create_port_for_buffered_design as create_port,
)
from elasticai.creator.hdl.translatable import Path
from elasticai.creator.hdl.vhdl.code_generation.template import Template


class FPLinear1d(Design):
    @property
    def port(self) -> Port:
        return self._port

    def __init__(
        self,
        *,
        in_feature_num: int,
        out_feature_num: int,
        total_bits: int,
        frac_bits: int,
        work_library_name: str = "work",
        resource_option: str = "auto",
        name: Optional[str] = None,
    ):
        super().__init__(
            name="fp_linear1d" if name is None else name,
        )
        self._port = create_port(
            x_width=total_bits,
            y_width=total_bits,
            x_count=in_feature_num,
            y_count=out_feature_num,
        )
        self.in_feature_num = in_feature_num
        self.out_feature_num = out_feature_num
        self.work_library_name = work_library_name
        self.resource_option = resource_option
        self.frac_width = frac_bits
        self.data_width = total_bits
        self.x_addr_width = self.port["x_address"].width
        self.y_addr_width = self.port["y_address"].width

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

    def save_to(self, destination: Path):
        template = Template(
            base_name="fp_linear_1d", package=module_to_package(self.__module__)
        )
        template.update_parameters(
            layer_name=self.name,
            work_library_name=self.work_library_name,
            resource_option=f'"{self.resource_option}"',
            **self._template_parameters(),
        )
        destination.as_file(f".vhd").write_text(template.lines())
