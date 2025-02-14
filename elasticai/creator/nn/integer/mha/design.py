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
        data_width: int,
        q_linear: object,
        k_linear: object,
        v_linear: object,
        inner_attn_module: object,
        output_linear: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._q_linear = q_linear
        self._k_linear = k_linear
        self._v_linear = v_linear
        self._inner_attn_module = inner_attn_module
        self._output_linear = output_linear

        self._work_library_name = work_library_name

        self.q_linear_design = self._q_linear.create_design(name=self._q_linear.name)
        self.k_linear_design = self._k_linear.create_design(name=self._k_linear.name)
        self.v_linear_design = self._v_linear.create_design(name=self._v_linear.name)
        self.inner_attn_module_design = self._inner_attn_module.create_design(
            name=self._inner_attn_module.name
        )
        self.output_linear_design = self._output_linear.create_design(
            name=self._output_linear.name
        )

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.q_linear_design._x_count,
            y_count=self.output_linear_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.q_linear_design.save_to(destination)
        self.k_linear_design.save_to(destination)
        self.v_linear_design.save_to(destination)
        self.inner_attn_module_design.save_to(destination)
        self.output_linear_design.save_to(destination)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="mha.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                q_linear_x_addr_width=str(self.q_linear_design._x_addr_width),
                q_linear_y_addr_width=str(self.q_linear_design._y_addr_width),
                k_linear_x_addr_width=str(self.k_linear_design._x_addr_width),
                k_linear_y_addr_width=str(self.k_linear_design._y_addr_width),
                v_linear_x_addr_width=str(self.v_linear_design._x_addr_width),
                v_linear_y_addr_width=str(self.v_linear_design._y_addr_width),
                inner_attn_y_address_width=str(
                    self.inner_attn_module_design._y_addr_width
                ),
                output_linear_y_addr_width=str(self.output_linear_design._y_addr_width),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="mha_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                q_linear_x_addr_width=str(self.q_linear_design._x_addr_width),
                q_linear_num_dimensions=str(self.q_linear_design._num_dimensions),
                q_linear_in_features=str(self.q_linear_design._in_features),
                k_linear_x_addr_width=str(self.k_linear_design._x_addr_width),
                k_linear_num_dimensions=str(self.k_linear_design._num_dimensions),
                k_linear_in_features=str(self.k_linear_design._in_features),
                v_linear_x_addr_width=str(self.v_linear_design._x_addr_width),
                v_linear_num_dimensions=str(self.v_linear_design._num_dimensions),
                v_linear_in_features=str(self.v_linear_design._in_features),
                output_linear_y_addr_width=str(self.output_linear_design._y_addr_width),
                output_linear_num_dimensions=str(
                    self.output_linear_design._num_dimensions
                ),
                output_linear_out_features=str(self.output_linear_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
