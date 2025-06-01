from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.LUT.design import LUT as LUTDesign
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.code_generation.addressable import calculate_address_width
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class ScaledDotProductAttention(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        matrix_multi_score: object,
        softmax: object,
        matrix_multi_att: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._matrix_multi_score = matrix_multi_score
        self._softmax = softmax
        self._matrix_multi_att = matrix_multi_att

        self._work_library_name = work_library_name

        self.matrix_multi_score_design = self._matrix_multi_score.create_design(
            name=self._matrix_multi_score.name
        )
        self.softmax_design = self._softmax.create_design(name=self._softmax.name)
        self.numerator_design = LUTDesign(
            name=self._softmax.name + "_numerator",
            x_data_width=self.softmax_design._data_width,
            y_data_width=self.softmax_design._numberator_lut_out_data_width,
            function=lambda x: self._softmax.Qinput2QNumerator_LUT_dict[x],
            inputs=list(self._softmax.Qinput2QNumerator_LUT_dict.keys()),
        )
        self.denominator_design = LUTDesign(
            name=self._softmax.name + "_denominator",
            x_data_width=self.softmax_design._data_width,
            y_data_width=self.softmax_design._denominator_lut_out_data_width,
            function=lambda x: self._softmax.Qinput2QDenominator_LUT_dict[x],
            inputs=list(self._softmax.Qinput2QDenominator_LUT_dict.keys()),
        )

        self.matrix_multi_att_design = self._matrix_multi_att.create_design(
            name=self._matrix_multi_att.name
        )

        self._x_1_addr_width = self.matrix_multi_score_design._x_1_addr_width
        self._x_2_addr_width = self.matrix_multi_score_design._x_2_addr_width
        self._y_score_addr_width = self.matrix_multi_score_design._y_addr_width

        self._x_3_addr_width = self.matrix_multi_att_design._x_2_addr_width
        self._y_addr_width = self.matrix_multi_att_design._y_addr_width

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.matrix_multi_score_design._x_count,
            y_count=self.matrix_multi_att_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.matrix_multi_score_design.save_to(
            destination.create_subpath(self._matrix_multi_score.name)
        )
        self.numerator_design.save_to(destination.create_subpath(self._softmax.name))
        self.denominator_design.save_to(destination.create_subpath(self._softmax.name))
        self.softmax_design.save_to(destination.create_subpath(self._softmax.name))
        self.matrix_multi_att_design.save_to(
            destination.create_subpath(self._matrix_multi_att.name)
        )

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="scaleddotproductattention.tpl.vhd",
            parameters=dict(
                name=self.name,
                matmul_score_name=self._matrix_multi_score.name,
                softmax_name=self._softmax.name,
                matmul_att_name=self._matrix_multi_att.name,
                data_width=str(self._data_width),
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                y_score_addr_width=str(self._y_score_addr_width),
                x_3_addr_width=str(self._x_3_addr_width),
                y_addr_width=str(self._y_addr_width),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="scaleddotproductattention_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                x_3_addr_width=str(self._x_3_addr_width),
                y_addr_width=str(self._y_addr_width),
                matmul_score_x_1_dim_a=str(self.matrix_multi_score_design._x_1_dim_a),
                matmul_score_x_1_dim_b=str(self.matrix_multi_score_design._x_1_dim_b),
                matmul_score_x_1_dim_c=str(self.matrix_multi_score_design._x_1_dim_c),
                matmul_score_x_2_dim_a=str(self.matrix_multi_score_design._x_2_dim_a),
                matmul_score_x_2_dim_b=str(self.matrix_multi_score_design._x_2_dim_b),
                matmul_score_x_2_dim_c=str(self.matrix_multi_score_design._x_2_dim_c),
                matmul_att_x_2_dim_a=str(self.matrix_multi_att_design._x_2_dim_a),
                matmul_att_x_2_dim_b=str(self.matrix_multi_att_design._x_2_dim_b),
                matmul_att_x_2_dim_c=str(self.matrix_multi_att_design._x_2_dim_c),
                matmul_att_y_dim_a=str(self.matrix_multi_att_design._y_dim_a),
                matmul_att_y_dim_b=str(self.matrix_multi_att_design._y_dim_b),
                matmul_att_y_dim_c=str(self.matrix_multi_att_design._y_dim_c),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
