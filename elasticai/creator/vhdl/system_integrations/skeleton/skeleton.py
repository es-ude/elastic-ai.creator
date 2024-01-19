from elasticai.creator.file_generation.savable import Path
from elasticai.creator.file_generation.template import (
    InProjectTemplate,
    module_to_package,
)
from elasticai.creator.vhdl.design.ports import Port


class Skeleton:
    def __init__(
        self,
        x_num_values: int,
        y_num_values: int,
        network_name: str,
        port: Port,
        id: int,
    ):
        self.name = "skeleton"
        self._network_name = network_name
        self._port = port
        self._x_num_values = str(x_num_values)
        self._y_num_values = str(y_num_values)
        self._id = id

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="network_skeleton.tpl.vhd",
            parameters=dict(
                name=self.name,
                network_name=self._network_name,
                data_width_in=str(self._port["x"].width),
                x_addr_width=str(self._port["x_address"].width),
                x_num_values=self._x_num_values,
                y_num_values=self._y_num_values,
                data_width_out=str(self._port["y"].width),
                y_addr_width=str(self._port["y_address"].width),
                id='x"{0:02x}"'.format(self._id),
            ),
        )
        file = destination.as_file(".vhd")
        file.write(template)


class LSTMSkeleton:
    def __init__(self, network_name: str):
        self.name = "skeleton"
        self._network_name = network_name

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name="lstm_network_skeleton.tpl.vhd",
            parameters=dict(name=self.name, network_name=self._network_name),
        )
        file = destination.as_file(".vhd")
        file.write(template)
