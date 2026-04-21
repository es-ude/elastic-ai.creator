from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.ram.design import Ram
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class SoftmaxLUT(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        numberator_lut_out_data_width: int,
        denominator_lut_out_data_width: int,
        dim_a: int,
        dim_b: int,
        dim_c: int,
        z_x: int,
        z_t: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        super().__init__(name=name)

        self._divider_name = f"{name}_divider"

        self._data_width = data_width
        self._numberator_lut_out_data_width = numberator_lut_out_data_width
        self._denominator_lut_out_data_width = denominator_lut_out_data_width
        self._dim_a = dim_a
        self._dim_b = dim_b
        self._dim_c = dim_c

        self._z_x = z_x
        self._z_t = z_t
        self._z_y = z_y

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_count = self._dim_a * self._dim_b * self._dim_c
        self._y_count = self._x_count

        self._x_addr_width = calculate_address_width(self._x_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="softmaxLUT.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                divider_name=str(self._divider_name),
                dim_a=str(self._dim_a),
                dim_b=str(self._dim_b),
                dim_c=str(self._dim_c),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                numberator_lut_out_data_width=str(self._numberator_lut_out_data_width),
                denominator_lut_out_data_width=str(
                    self._denominator_lut_out_data_width
                ),
                z_x=str(self._z_x),
                z_t=str(self._z_t),
                z_y=str(self._z_y),
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        divider_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="Divider.tpl.vhd",
            parameters=dict(
                name=self._divider_name,
                data_width=str(self._numberator_lut_out_data_width + 1),
            ),
        )
        destination.create_subpath(self._divider_name).as_file(".vhd").write(
            divider_template
        )

        ram = Ram(name=f"{self.name}_ram")
        ram.save_to(destination)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="softmaxLUT_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                dim_a=str(self._dim_a),
                dim_b=str(self._dim_b),
                dim_c=str(self._dim_c),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                numberator_lut_out_data_width=str(self._numberator_lut_out_data_width),
                denominator_lut_out_data_width=str(
                    self._denominator_lut_out_data_width
                ),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
