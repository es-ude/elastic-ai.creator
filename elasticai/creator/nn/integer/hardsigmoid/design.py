from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class HardSigmoid(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        quantized_three: int,
        quantized_minus_three: int,
        quantized_one: int,
        quantized_zero: int,
        tmp: int,  # Note: at the moment hardware side only supports 16 bits signed of TEMP.
        work_library_name: str,
        template_variant: str = "lut_step_function",  # default, lut_1to1_mapping
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._work_library_name = work_library_name

        self._quantized_three = quantized_three
        self._quantized_minus_three = quantized_minus_three
        self._quantized_one = quantized_one
        self._quantized_zero = quantized_zero
        self._tmp = tmp
        self._template_variant = template_variant

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
        )

    def save_to(self, destination: Path) -> None:
        if self._template_variant == "default":
            template = InProjectTemplate(
                package=module_to_package(self.__module__),
                file_name="hardsigmoid.tpl.vhd",
                parameters=dict(
                    name=self.name,
                    data_width=str(self._data_width),
                    three_threshold=str(self._quantized_three),
                    minus_three_threshold=str(self._quantized_minus_three),
                    zero_output=str(self._quantized_zero),
                    one_output=str(self._quantized_one),
                    tmp_threshold=str(self._tmp),
                    work_library_name=self._work_library_name,
                ),
            )
        elif self._template_variant == "lut_1to1_mapping":
            template = self._gen_1to1_mapping_lut_implementation()
        elif self._template_variant == "lut_step_function":
            template = self._gen_step_lut_implementation()
        else:
            raise ValueError(f"Unsupported template variant: {self._template_variant}")

        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="hardsigmoid_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                work_library_name=self._work_library_name,
            ),
        )

        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )

    def _gen_1to1_mapping_lut_implementation(self) -> None:
        """Generate 1 to 1 mapping LUT implementation for HardSigmoid."""

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="LUT.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_data_width=str(self._data_width),
                y_data_width=str(self._data_width),
            ),
        )

        process_content = []

        process_content.append(
            f"if signed_x <= {self._quantized_minus_three} then "
            f"signed_y <= to_signed({self._quantized_zero}, {self._data_width});"
        )
        for input_value in range(
            self._quantized_minus_three + 1, self._quantized_three - 1
        ):
            # Calculate the corresponding output value based on the HardSigmoid formula
            output_value = int((input_value) / 8.0) + self._tmp
            process_content.append(
                f"elsif signed_x = {input_value} then "
                f"signed_y <= to_signed({output_value}, {self._data_width});"
            )

        process_content.append(
            f"else signed_y <= to_signed({self._quantized_one}, {self._data_width});"
        )
        process_content.append("end if;")

        self._process_content = process_content

        template.parameters.update(process_content=process_content)

        return template

    def _gen_step_lut_implementation(self) -> None:
        """Generate step function based LUT implementation for HardSigmoid."""

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="LUT.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_data_width=str(self._data_width),
                y_data_width=str(self._data_width),
            ),
        )

        process_content = []

        process_content.append(
            f"if signed_x <= {self._quantized_minus_three} then "
            f"signed_y <= to_signed({self._quantized_zero}, {self._data_width});"
        )
        for input_value in range(
            self._quantized_minus_three + 1, self._quantized_three - 1
        ):
            # Calculate the corresponding output value based on the HardSigmoid formula
            output_value = int((input_value) / 8.0) + self._tmp
            next_output_value = int((input_value + 1) / 8.0) + self._tmp
            if output_value != next_output_value:
                process_content.append(
                    f"elsif signed_x <= {input_value} then "
                    f"signed_y <= to_signed({output_value}, {self._data_width});"
                )
            elif input_value == self._quantized_three - 2:
                process_content.append(
                    f"elsif signed_x <= {input_value} then "
                    f"signed_y <= to_signed({output_value}, {self._data_width});"
                )

        process_content.append(
            f"else signed_y <= to_signed({self._quantized_one}, {self._data_width});"
        )
        process_content.append("end if;")

        self._process_content = process_content

        template.parameters.update(process_content=process_content)

        return template
