import numpy as np

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


class AVGPooling1dFlatten(Design):
    """VHDL design of an integer global average pooling with flattening.

    For every channel, sums the `in_features` inputs of the sequence, scales
    the sum by `m_q * 2**-m_q_shift` and adds the output zero point. The
    layer uses the buffered interface (`x`, `x_address`, `y`, `y_address`,
    `enable`, `done`, `clock`).
    """

    def __init__(
        self,
        name: str,
        data_width: int,
        in_features: int,
        out_features: int,
        in_num_dimensions: int,
        out_num_dimensions: int,
        m_q: int,
        m_q_shift: int,
        z_x: int,
        z_y: int,
        work_library_name: str,
        resource_option: str,
    ) -> None:
        """Create the design.

        Args:
            name: Name of the VHDL entity.
            data_width: Bit width of inputs and outputs.
            in_features: Sequence length of the inputs.
            out_features: Sequence length of the outputs, usually 1.
            in_num_dimensions: Number of channels of the inputs.
            out_num_dimensions: Number of channels of the outputs.
            m_q: Integer multiplier of the scale `M`.
            m_q_shift: Right shift of the scale `M`.
            z_x: Zero point of the inputs.
            z_y: Zero point of the outputs.
            work_library_name: Name of the VHDL library the design is
                compiled into.
            resource_option: Resource type used for the output RAM.
        """
        super().__init__(name=name)

        self._data_width = data_width
        self._in_features = in_features
        self._out_features = out_features
        self._in_num_dimensions = in_num_dimensions
        self._out_num_dimensions = out_num_dimensions

        self._m_q = m_q
        self._m_q_shift = m_q_shift
        self._m_q_data_width = (
            int(np.ceil(np.log2(self._m_q))) + 1
            if self._m_q != 0
            else 1
            if self._m_q != 0
            else 1
        )

        self._z_x = z_x
        self._z_y = z_y

        self._work_library_name = work_library_name
        self._resource_option = resource_option

        self._x_count = self._in_features * self._in_num_dimensions
        self._y_count = self._out_features * self._out_num_dimensions

        self._x_addr_width = calculate_address_width(self._x_count)
        self._y_addr_width = calculate_address_width(self._y_count)

    @property
    def port(self) -> Port:
        """Buffered port with `x_count` inputs and `y_count` outputs."""
        return create_port(
            x_width=self._data_width,
            y_width=self._data_width,
            x_count=self._x_count,
            y_count=self._y_count,
        )

    def save_to(self, destination: Path) -> None:
        """Write the layer, its output RAM and a testbench.

        Args:
            destination: Directory the VHDL files are written to.
        """
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="avgpooling1dflatten.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                in_num_dimensions=str(self._in_num_dimensions),
                out_num_dimensions=str(self._out_num_dimensions),
                m_q_data_width=str(self._m_q_data_width),
                z_x=str(self._z_x),
                z_y=str(self._z_y),
                m_q=str(self._m_q),
                m_q_shift=str(self._m_q_shift),
                work_library_name=self._work_library_name,
                resource_option=self._resource_option,
            ),
        )
        destination.create_subpath(self.name).as_file(".vhd").write(template)

        ram = Ram(name=f"{self.name}_ram")
        ram.save_to(destination)

        template_test = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="avgpooling1dflatten_tb.tpl.vhd",
            parameters=dict(
                name=self.name,
                x_addr_width=str(self._x_addr_width),
                y_addr_width=str(self._y_addr_width),
                data_width=str(self._data_width),
                in_features=str(self._in_features),
                out_features=str(self._out_features),
                in_num_dimensions=str(self._in_num_dimensions),
                out_num_dimensions=str(self._out_num_dimensions),
                work_library_name=self._work_library_name,
            ),
        )
        destination.create_subpath(f"{self.name}_tb").as_file(".vhd").write(
            template_test
        )
