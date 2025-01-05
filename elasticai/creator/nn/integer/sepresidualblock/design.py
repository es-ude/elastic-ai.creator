from itertools import chain

from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.auto_wire_protocols.port_definitions import create_port
from elasticai.creator.vhdl.design.design import Design


class SeparableResidualBlock(Design):
    def __init__(
        self,
        name: str,
        data_width: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weights_depthwise: list[list[int]],
        weights_pointwise1: list[list[int]],
        weights_pointwise2: list[list[int]],
        bias_depthwise: list[int],
        bias_pointwise1: list[int],
        bias_pointwise2: list[int],
        m_q: int,
        m_q_shift: int,
        z_x: int,
        z_w: int,
        z_b: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        super().__init__(name=name)

        self._data_width = data_width
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size

        self._weights_depthwise = weights_depthwise
        self._weights_pointwise1 = weights_pointwise1
        self._weights_pointwise2 = weights_pointwise2

        self._bias_depthwise = bias_depthwise
        self._bias_pointwise1 = bias_pointwise1
        self._bias_pointwise2 = bias_pointwise2

        self._m_q = m_q
        self._m_q_shift = m_q_shift

        self._z_x = z_x
        self._z_w = z_w
        self._z_b = z_b
        self._z_y = z_y

        self._work_library_name = work_library_name
        self._resource_option = resource_option

    @property
    def port(self):
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._in_channels,
            y_count=self._out_channels,
        )

    def save_to(self, destination: Path) -> None:
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="sepresidualblock.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=self._data_width,
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=self._kernel_size,
                m_q=self._m_q,
                m_q_shift=self._m_q_shift,
                z_x=self._z_x,
                z_w=self._z_w,
                z_b=self._z_b,
                z_y=self._z_y,
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        test_template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="sepresidualblock_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                data_width=self._data_width,
                in_channels=self._in_channels,
                out_channels=self._out_channels,
                kernel_size=self._kernel_size,
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            test_template
        )
