from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class EncoderLayer(Design):
    def __init__(
        self,
        name: str,
        mha: object,
        mha_add: object,
        mha_norm: object,
        ffn: object,
        ffn_add: object,
        ffn_norm: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._work_library_name = work_library_name
        self._mha_name = mha.name
        self._mha_add_name = mha_add.name
        self._mha_norm_name = mha_norm.name
        self._ffn_name = ffn.name
        self._ffn_add_name = ffn_add.name
        self._ffn_norm_name = ffn_norm.name

        self._mha_design = mha.create_design(name=self._mha_name)
        self._mha_add_design = mha_add.create_design(name=self._mha_add_name)
        self._mha_norm_design = mha_norm.create_design(name=self._mha_norm_name)
        self._ffn_design = ffn.create_design(name=self._ffn_name)
        self._ffn_add_design = ffn_add.create_design(name=self._ffn_add_name)
        self._ffn_norm_design = ffn_norm.create_design(name=self._ffn_norm_name)

        self._x_data_width = self._mha_design._x_1_data_width
        self._y_data_width = self._ffn_norm_design._y_data_width

        self._x_addr_width = self._mha_design._x_1_addr_width
        self._y_addr_width = self._ffn_norm_design._y_addr_width

        self._x_count = self._mha_design._x_1_count
        self._y_count = self._ffn_norm_design._y_count

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._x_data_width,
            y_width=self._y_data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self._mha_design.save_to(destination.create_subpath(self._mha_name))
        self._mha_add_design.save_to(destination.create_subpath(self._mha_add_name))
        self._mha_norm_design.save_to(destination.create_subpath(self._mha_norm_name))
        self._ffn_design.save_to(destination.create_subpath(self._ffn_name))
        self._ffn_add_design.save_to(destination.create_subpath(self._ffn_add_name))
        self._ffn_norm_design.save_to(destination.create_subpath(self._ffn_norm_name))

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="encoderlayer.tpl.vhd",
            parameters=dict(
                name=self.name,
                mha_name=self._mha_name,
                mha_add_name=self._mha_add_name,
                mha_norm_name=self._mha_norm_name,
                ffn_name=self._ffn_name,
                ffn_add_name=self._ffn_add_name,
                ffn_norm_name=self._ffn_norm_name,
                work_library_name=self._work_library_name,
                x_data_width=str(self._x_data_width),
                mha_y_data_width=str(self._mha_design._y_data_width),
                mha_add_x_1_data_width=str(self._mha_add_design._x_1_data_width),
                mha_add_x_2_data_width=str(self._mha_add_design._x_2_data_width),
                mha_add_y_data_width=str(self._mha_add_design._y_data_width),
                mha_norm_x_data_width=str(self._mha_norm_design._x_data_width),
                mha_norm_y_data_width=str(self._mha_norm_design._y_data_width),
                ffn_x_data_width=str(self._ffn_design._x_data_width),
                ffn_y_data_width=str(self._ffn_design._y_data_width),
                ffn_add_x_1_data_width=str(self._ffn_add_design._x_1_data_width),
                ffn_add_x_2_data_width=str(self._ffn_add_design._x_2_data_width),
                ffn_add_y_data_width=str(self._ffn_add_design._y_data_width),
                ffn_norm_x_data_width=str(self._ffn_norm_design._x_data_width),
                y_data_width=str(self._y_data_width),
                x_addr_width=str(self._x_addr_width),
                mha_y_addr_width=str(self._mha_design._y_addr_width),
                mha_add_y_addr_width=str(self._mha_add_design._y_addr_width),
                mha_norm_y_addr_width=str(self._mha_norm_design._y_addr_width),
                ffn_y_addr_width=str(self._ffn_design._y_addr_width),
                ffn_add_y_addr_width=str(self._ffn_add_design._y_addr_width),
                y_addr_width=str(self._y_addr_width),
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="encoderlayer_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                work_library_name=self._work_library_name,
                x_data_width=str(self._x_data_width),
                y_data_width=str(self._y_data_width),
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                x_count=str(self._x_count),
                y_count=str(self._y_count),
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
