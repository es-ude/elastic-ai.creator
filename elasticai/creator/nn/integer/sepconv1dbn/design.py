from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design
from elasticai.creator.vhdl.design.ports import Port


class SepConv1dBN(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        depthconv1d: object,
        pointconv1dbn: object,
        work_library_name: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._depthconv1d = depthconv1d
        self._pointconv1dbn = pointconv1dbn

        self._work_library_name = work_library_name

        self.depthconv1d_design = self._depthconv1d.create_design(
            name=self._depthconv1d.name
        )
        self.pointconv1dbn_design = self._pointconv1dbn.create_design(
            name=self._pointconv1dbn.name
        )

        self._x_count = self.depthconv1d_design._x_count
        self._y_count = self.pointconv1dbn_design._y_count

        self._x_addr_width = self.depthconv1d_design._x_addr_width
        self._y_addr_width = self.pointconv1dbn_design._y_addr_width

    @property
    def port(self) -> Port:
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        self.depthconv1d_design.save_to(
            destination.create_subpath(self._depthconv1d.name)
        )
        self.pointconv1dbn_design.save_to(
            destination.create_subpath(self._pointconv1dbn.name)
        )

        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="sepconv1dbn.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=str(self.depthconv1d_design._x_addr_width),
                depth_y_addr_width=str(self.depthconv1d_design._y_addr_width),
                y_addr_width=str(self.pointconv1dbn_design._y_addr_width),
                data_width=str(self._data_width),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="sepconv1dbn_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=str(self._data_width),
                x_addr_width=str(self.depthconv1d_design._x_addr_width),
                y_addr_width=str(self.pointconv1dbn_design._y_addr_width),
                in_channels=str(self.depthconv1d_design._in_channels),
                seq_len=str(self.depthconv1d_design._seq_len),
                kernel_size=str(self.depthconv1d_design._kernel_size),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(self.name).as_file("_tb.vhd").write(template_test)


def _flatten_params(params):
    flat_list = []
    for p in params:
        if isinstance(p, list):
            flat_list.extend(_flatten_params(p))
        else:
            flat_list.append(p)
    return flat_list
