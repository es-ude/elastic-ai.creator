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
        data_width: int,
        mha: object,
        mha_add: object,
        mha_norm: object,
        ffn: object,
        ffn_add: object,
        ffn_norm: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._mha = mha
        self._mha_add = mha_add
        self._mha_norm = mha_norm
        self._ffn = ffn
        self._ffn_add = ffn_add
        self._ffn_norm = ffn_norm

        self._work_library_name = work_library_name

        self.mha_design = self._mha.create_design(name=self._mha.name)
        self.mha_add_design = self._mha_add.create_design(name=self._mha_add.name)
        self.mha_norm_design = self._mha_norm.create_design(name=self._mha_norm.name)
        self.ffn_design = self._ffn.create_design(name=self._ffn.name)
        self.ffn_add_design = self._ffn_add.create_design(name=self._ffn_add.name)
        self.ffn_norm_design = self._ffn_norm.create_design(name=self._ffn_norm.name)

        self._x_addr_width = self.mha_design._x_addr_width
        self._num_dimensions = self.mha_design._num_dimensions
        self._in_features = self.mha_design._in_features
        self._y_addr_width = self.ffn_norm_design._y_addr_width
        self._out_features = self.ffn_norm_design._out_features

        self._x_count = self.mha_design._x_count
        self._y_count = self.ffn_norm_design._y_count

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self.mha_design._x_count,
            y_count=self.ffn_norm_design._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.mha_design.save_to(destination.create_subpath(self._mha.name))
        self.mha_add_design.save_to(destination.create_subpath(self._mha_add.name))
        self.mha_norm_design.save_to(destination.create_subpath(self._mha_norm.name))
        self.ffn_design.save_to(destination.create_subpath(self._ffn.name))
        self.ffn_add_design.save_to(destination.create_subpath(self._ffn_add.name))
        self.ffn_norm_design.save_to(destination.create_subpath(self._ffn_norm.name))

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="encoderlayer.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.mha_design._x_addr_width),
                y_addr_width=str(self.ffn_norm_design._y_addr_width),
                num_dimensions=str(self.mha_design._num_dimensions),
                in_features=str(self.mha_design._in_features),
                out_features=str(self.mha_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="encoderlayer_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.mha_design._x_addr_width),
                y_addr_width=str(self.ffn_norm_design._y_addr_width),
                num_dimensions=str(self.mha_design._num_dimensions),
                in_features=str(self.mha_design._in_features),
                out_features=str(self.mha_design._out_features),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)
