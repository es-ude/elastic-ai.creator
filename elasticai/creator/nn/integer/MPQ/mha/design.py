from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class MHA(Design):
    def __init__(
        self,
        name: str,
        q_linear: object,
        k_linear: object,
        v_linear: object,
        inner_attn_module: object,
        output_linear: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._work_library_name = work_library_name
        self._q_linear_name = q_linear.name
        self._k_linear_name = k_linear.name
        self._v_linear_name = v_linear.name
        self._inner_attn_module_name = inner_attn_module.name
        self._output_linear_name = output_linear.name

        self._q_linear_design = q_linear.create_design(name=self._q_linear_name)
        self._k_linear_design = k_linear.create_design(name=self._k_linear_name)
        self._v_linear_design = v_linear.create_design(name=self._v_linear_name)
        self._inner_attn_module_design = inner_attn_module.create_design(
            name=self._inner_attn_module_name
        )
        self.output_linear_design = output_linear.create_design(
            name=self._output_linear_name
        )

        self._x_1_data_width = self._q_linear_design._x_data_width
        self._x_2_data_width = self._k_linear_design._x_data_width
        self._x_3_data_width = self._v_linear_design._x_data_width
        self._y_data_width = self.output_linear_design._y_data_width

        self._x_1_addr_width = self._q_linear_design._x_addr_width
        self._x_2_addr_width = self._k_linear_design._x_addr_width
        self._x_3_addr_width = self._v_linear_design._x_addr_width
        self._y_addr_width = self.output_linear_design._y_addr_width

        self._x_1_count = self._q_linear_design._x_count
        self._x_2_count = self._k_linear_design._x_count
        self._x_3_count = self._v_linear_design._x_count
        self._y_count = self.output_linear_design._y_count

    @property
    def port(self) -> Port:
        return create_port(
            x_1_width=self._x_1_data_width,
            x_2_width=self._x_2_data_width,
            x_3_width=self._x_3_data_width,
            y_width=self._y_data_width,
            x_1_count=self._x_1_count,
            x_2_count=self._x_2_count,
            x_3_count=self._x_3_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self._q_linear_design.save_to(destination.create_subpath(self._q_linear_name))
        self._k_linear_design.save_to(destination.create_subpath(self._k_linear_name))
        self._v_linear_design.save_to(destination.create_subpath(self._v_linear_name))
        self.inner_attn_module_design.save_to(
            destination.create_subpath(self._inner_attn_module_name)
        )
        self.output_linear_design.save_to(
            destination.create_subpath(self._output_linear_name)
        )

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="mha.tpl.vhd",
            parameters=dict(
                name=self.name,
                q_linear_name=self._q_linear_name,
                k_linear_name=self._k_linear_name,
                v_linear_name=self._v_linear_name,
                inner_attn_module_name=self._inner_attn_module_name,
                output_linear_name=self._output_linear_name,
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                x_3_addr_width=str(self._x_3_addr_width),
                q_linear_y_addr_width=str(self._q_linear_design._y_addr_width),
                k_linear_y_addr_width=str(self._k_linear_design._y_addr_width),
                v_linear_y_addr_width=str(self._v_linear_design._y_addr_width),
                inner_attn_x_1_addr_width=str(
                    self.inner_attn_module_design._x_1_addr_width
                ),
                inner_attn_x_2_addr_width=str(
                    self.inner_attn_module_design._x_2_addr_width
                ),
                inner_attn_x_3_addr_width=str(
                    self.inner_attn_module_design._x_3_addr_width
                ),
                inner_attn_y_addr_width=str(
                    self.inner_attn_module_design._y_addr_width
                ),
                output_linear_y_addr_width=str(self.output_linear_design._y_addr_width),
                y_addr_width=str(self._y_addr_width),
                x_1_data_width=str(self._x_1_data_width),
                x_2_data_width=str(self._x_2_data_width),
                x_3_data_width=str(self._x_3_data_width),
                q_linear_y_data_width=str(self._q_linear_design._y_data_width),
                k_linear_y_data_width=str(self._k_linear_design._y_data_width),
                v_linear_y_data_width=str(self._v_linear_design._y_data_width),
                inner_attn_x_1_data_width=str(
                    self.inner_attn_module_design._x_1_data_width
                ),
                inner_attn_x_2_data_width=str(
                    self.inner_attn_module_design._x_2_data_width
                ),
                inner_attn_x_3_data_width=str(
                    self.inner_attn_module_design._x_3_data_width
                ),
                inner_attn_y_data_width=str(
                    self.inner_attn_module_design._y_data_width
                ),
                output_linear_x_data_width=str(self.output_linear_design._x_data_width),
                y_data_width=str(self._y_data_width),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="mha_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_1_data_width=str(self._x_1_data_width),
                x_2_data_width=str(self._x_2_data_width),
                x_3_data_width=str(self._x_3_data_width),
                y_data_width=str(self._y_data_width),
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                x_3_addr_width=str(self._x_3_addr_width),
                y_addr_width=str(self._y_addr_width),
                x_1_count=str(self._x_1_count),
                x_2_count=str(self._x_2_count),
                x_3_count=str(self._x_3_count),
                y_count=str(self._y_count),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
