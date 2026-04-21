from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class GRUCell(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        concatenate: object,
        f_gate_linear: object,
        c_gate_linear: object,
        i_gate_linear: object,
        o_gate_linear: object,
        i_sigmoid: object,
        f_sigmoid: object,
        o_sigmoid: object,
        c_tanh: object,
        c_next_tanh: object,
        c_next_addition: object,
        fc_hadamard_product: object,
        ic_hadamard_product: object,
        oc_hadamard_product: object,
        work_library_name: str,
    ):
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._concatenate = concatenate
        self._f_gate_linear = f_gate_linear
        self._c_gate_linear = c_gate_linear
        self._i_gate_linear = i_gate_linear
        self._o_gate_linear = o_gate_linear
        self._i_sigmoid = i_sigmoid
        self._f_sigmoid = f_sigmoid
        self._o_sigmoid = o_sigmoid
        self._c_tanh = c_tanh
        self._c_next_tanh = c_next_tanh
        self._c_next_addition = c_next_addition
        self._fc_hadamard_product = fc_hadamard_product
        self._ic_hadamard_product = ic_hadamard_product
        self._oc_hadamard_product = oc_hadamard_product

        self.concatenate_design = self._concatenate.create_design(
            name=self._concatenate.name
        )
        self.f_gate_linear_design = self._f_gate_linear.create_design(
            name=self._f_gate_linear.name
        )
        self.c_gate_linear_design = self._c_gate_linear.create_design(
            name=self._c_gate_linear.name
        )
        self.i_gate_linear_design = self._i_gate_linear.create_design(
            name=self._i_gate_linear.name
        )
        self.o_gate_linear_design = self._o_gate_linear.create_design(
            name=self._o_gate_linear.name
        )
        self.i_sigmoid_design = self._i_sigmoid.create_design(name=self._i_sigmoid.name)
        self.f_sigmoid_design = self._f_sigmoid.create_design(name=self._f_sigmoid.name)
        self.o_sigmoid_design = self._o_sigmoid.create_design(name=self._o_sigmoid.name)
        self.c_tanh_design = self._c_tanh.create_design(name=self._c_tanh.name)
        self.c_next_tanh_design = self._c_next_tanh.create_design(
            name=self._c_next_tanh.name
        )
        self.c_next_addition_design = self._c_next_addition.create_design(
            name=self._c_next_addition.name
        )
        self.fc_hadamard_product_design = self._fc_hadamard_product.create_design(
            name=self._fc_hadamard_product.name
        )
        self.ic_hadamard_product_design = self._ic_hadamard_product.create_design(
            name=self._ic_hadamard_product.name
        )
        self.oc_hadamard_product_design = self._oc_hadamard_product.create_design(
            name=self._oc_hadamard_product.name
        )

    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.concatenate_design._x_count,
            y_count=self.oc_hadamard_product_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.concatenate_design.save_to(destination)
        self.f_gate_linear_design.save_to(destination)
        self.c_gate_linear_design.save_to(destination)
        self.i_gate_linear_design.save_to(destination)
        self.o_gate_linear_design.save_to(destination)
        self.i_sigmoid_design.save_to(destination)
        self.f_sigmoid_design.save_to(destination)
        self.o_sigmoid_design.save_to(destination)
        self.c_tanh_design.save_to(destination)
        self.c_next_tanh_design.save_to(destination)
        self.c_next_addition_design.save_to(destination)
        self.fc_hadamard_product_design.save_to(destination)
        self.ic_hadamard_product_design.save_to(destination)
        self.oc_hadamard_product_design.save_to(destination)

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstmcell.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                concatenate_x_1_addr_width=str(self.concatenate_design._x_1_addr_width),
                concatenate_x_2_addr_width=str(self.concatenate_design._x_2_addr_width),
                concatenate_y_addr_width=str(self.concatenate_design._y_addr_width),
                f_gate_linear_x_addr_width=str(self.f_gate_linear_design._x_addr_width),
                f_gate_linear_y_addr_width=str(self.f_gate_linear_design._y_addr_width),
                c_gate_linear_x_addr_width=str(self.c_gate_linear_design._x_addr_width),
                c_gate_linear_y_addr_width=str(self.c_gate_linear_design._y_addr_width),
                i_gate_linear_x_addr_width=str(self.i_gate_linear_design._x_addr_width),
                i_gate_linear_y_addr_width=str(self.i_gate_linear_design._y_addr_width),
                o_gate_linear_x_addr_width=str(self.o_gate_linear_design._x_addr_width),
                o_gate_linear_y_addr_width=str(self.o_gate_linear_design._y_addr_width),
                i_sigmoid_x_addr_width=str(self.i_sigmoid_design._x_addr_width),
                i_sigmoid_y_addr_width=str(self.i_sigmoid_design._y_addr_width),
                f_sigmoid_x_addr_width=str(self.f_sigmoid_design._x_addr_width),
                f_sigmoid_y_addr_width=str(self.f_sigmoid_design._y_addr_width),
                o_sigmoid_x_addr_width=str(self.o_sigmoid_design._x_addr_width),
                o_sigmoid_y_addr_width=str(self.o_sigmoid_design._y_addr_width),
                c_tanh_x_addr_width=str(self.c_tanh_design._x_addr_width),
                c_tanh_y_addr_width=str(self.c_tanh_design._y_addr_width),
                c_next_tanh_x_addr_width=str(self.c_next_tanh_design._x_addr_width),
                c_next_tanh_y_addr_width=str(self.c_next_tanh_design._y_addr_width),
                c_next_addition_x_1_addr_width=str(
                    self.c_next_addition_design._x_1_addr_width
                ),
                c_next_addition_x_2_addr_width=str(
                    self.c_next_addition_design._x_2_addr_width
                ),
                c_next_addition_y_addr_width=str(
                    self.c_next_addition_design._y_addr_width
                ),
                fc_hadamard_product_x_1_addr_width=str(
                    self.fc_hadamard_product_design._x_1_addr_width
                ),
                fc_hadamard_product_x_2_addr_width=str(
                    self.fc_hadamard_product_design._x_2_addr_width
                ),
                fc_hadamard_product_y_addr_width=str(
                    self.fc_hadamard_product_design._y_addr_width
                ),
                ic_hadamard_product_x_1_addr_width=str(
                    self.ic_hadamard_product_design._x_1_addr_width
                ),
                ic_hadamard_product_x_2_addr_width=str(
                    self.ic_hadamard_product_design._x_2_addr_width
                ),
                ic_hadamard_product_y_addr_width=str(
                    self.ic_hadamard_product_design._y_addr_width
                ),
                oc_hadamard_product_x_1_addr_width=str(
                    self.oc_hadamard_product_design._x_1_addr_width
                ),
                oc_hadamard_product_x_2_addr_width=str(
                    self.oc_hadamard_product_design._x_2_addr_width
                ),
                oc_hadamard_product_y_addr_width=str(
                    self.oc_hadamard_product_design._y_addr_width
                ),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstmcell_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_1_addr_width=str(self.concatenate_design._x_1_addr_width),
                x_2_addr_width=str(self.concatenate_design._x_2_addr_width),
                x_3_addr_width=str(self.fc_hadamard_product_design._x_1_addr_width),
                y_1_addr_width=str(self.oc_hadamard_product_design._y_addr_width),
                y_2_addr_width=str(self.fc_hadamard_product_design._y_addr_width),
                x1_num_features=str(self.concatenate_design._x1_num_features),
                x1_num_dimensions=str(self.concatenate_design._x1_num_dimensions),
                x2_num_features=str(self.concatenate_design._x2_num_features),
                x2_num_dimensions=str(self.concatenate_design._x2_num_dimensions),
                x_3_num_features=str(self.fc_hadamard_product_design._x_1_num_features),
                x_3_num_dimensions=str(
                    self.fc_hadamard_product_design._x_1_num_dimensions
                ),
                y_1_num_features=str(self.oc_hadamard_product_design._y_num_features),
                y_1_num_dimensions=str(
                    self.oc_hadamard_product_design._y_num_dimensions
                ),
                y_2_num_features=str(self.fc_hadamard_product_design._y_num_features),
                y_2_num_dimensions=str(
                    self.fc_hadamard_product_design._y_num_dimensions
                ),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
