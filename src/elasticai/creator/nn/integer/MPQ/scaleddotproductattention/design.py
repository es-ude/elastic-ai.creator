from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.nn.integer.LUT.design import LUT as LUTDesign
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class ScaledDotProductAttention(Design):
    def __init__(
        self,
        name: str,
        matrix_multi_score: object,
        softmax: object,
        matrix_multi_att: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._work_library_name = work_library_name
        self._matrix_multi_score_name = matrix_multi_score.name
        self._softmax_name = softmax.name
        self._matrix_multi_att_name = matrix_multi_att.name

        self._matrix_multi_score_design = matrix_multi_score.create_design(
            name=matrix_multi_score.name
        )
        self._softmax_design = softmax.create_design(name=softmax.name)
        self._numerator_design = LUTDesign(
            name=self._softmax_name + "_numerator",
            x_data_width=self._softmax_design._x_data_width,
            y_data_width=self._softmax_design._numberator_lut_out_data_width,
            function=lambda x: softmax.Qinput2QNumerator_LUT_dict[x],
            inputs=list(softmax.Qinput2QNumerator_LUT_dict.keys()),
        )
        self._denominator_design = LUTDesign(
            name=self._softmax_name + "_denominator",
            x_data_width=self._softmax_design._x_data_width,
            y_data_width=self._softmax_design._denominator_lut_out_data_width,
            function=lambda x: softmax.Qinput2QDenominator_LUT_dict[x],
            inputs=list(softmax.Qinput2QDenominator_LUT_dict.keys()),
        )
        self._matrix_multi_att_design = matrix_multi_att.create_design(
            name=self._matrix_multi_att_name
        )

        self._x_1_data_width = self._matrix_multi_score_design._x_1_data_width
        self._x_2_data_width = self._matrix_multi_score_design._x_2_data_width
        self._x_3_data_width = self._matrix_multi_att_design._x_2_data_width
        self._y_data_width = self._matrix_multi_att_design._y_data_width

        self._x_1_addr_width = self._matrix_multi_score_design._x_1_addr_width
        self._x_2_addr_width = self._matrix_multi_score_design._x_2_addr_width
        self._x_3_addr_width = self._matrix_multi_att_design._x_2_addr_width
        self._y_addr_width = self._matrix_multi_att_design._y_addr_width

        self._x_1_count = self._matrix_multi_score_design._x_1_count
        self._x_2_count = self._matrix_multi_score_design._x_2_count
        self._x_3_count = self._matrix_multi_att_design._x_2_count
        self._y_count = self._matrix_multi_att_design._y_count

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
        self._matrix_multi_score_design.save_to(
            destination.create_subpath(self._matrix_multi_score_name)
        )
        self._numerator_design.save_to(destination.create_subpath(self._softmax_name))
        self._denominator_design.save_to(destination.create_subpath(self._softmax_name))
        self._softmax_design.save_to(destination.create_subpath(self._softmax_name))
        self._matrix_multi_att_design.save_to(
            destination.create_subpath(self._matrix_multi_att_name)
        )

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="scaleddotproductattention.tpl.vhd",
            parameters=dict(
                name=self.name,
                matmul_score_name=self._matrix_multi_score_name,
                softmax_name=self._softmax_name,
                matmul_att_name=self._matrix_multi_att_name,
                x_1_data_width=str(self._x_1_data_width),
                x_2_data_width=str(self._x_2_data_width),
                matrix_multi_score_y_data_width=str(
                    self._matrix_multi_score_design._y_data_width
                ),
                softmax_x_data_width=str(self._softmax_design._x_data_width),
                softmax_y_data_width=str(self._softmax_design._y_data_width),
                matrix_multi_att_x_1_data_width=str(
                    self._matrix_multi_att_design._x_1_data_width
                ),
                x_3_data_width=str(self._x_3_data_width),
                y_data_width=str(self._y_data_width),
                x_1_addr_width=str(self._x_1_addr_width),
                x_2_addr_width=str(self._x_2_addr_width),
                x_3_addr_width=str(self._x_3_addr_width),
                matrix_multi_score_y_addr_width=str(
                    self._matrix_multi_score_design._y_addr_width
                ),
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
