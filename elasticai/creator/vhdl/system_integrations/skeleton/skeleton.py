import warnings

from elasticai.creator.arithmetic import FxpConverter, FxpParams
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
        id: list[int] | int,
        skeleton_version: str = "v1",
    ):
        self.name = "skeleton"
        self._network_name = network_name
        self._port = port
        self._x_num_values = str(x_num_values)
        self._y_num_values = str(y_num_values)
        if isinstance(id, int):
            id = [id]
        self._id = id
        if skeleton_version == "v1":
            warnings.warn(
                (
                    "Skeleton V1 might be deprecated in the future. Consider using"
                    " Skeleton V2 instead."
                ),
                FutureWarning,
            )
            self._template_file_name = "network_skeleton.tpl.vhd"
            if len(id) != 1:
                raise Exception(
                    f"should give an id of 1 byte. Actual length is {len(id)}"
                )
            if x_num_values > 100:
                raise Exception(
                    "Not more than 100 input values allowed. Actual num of inputs"
                    f" {x_num_values} ."
                )
            if y_num_values > 100:
                raise Exception(
                    "Not more than 100 input values allowed. Actual num of inputs"
                    f" {x_num_values} ."
                )
        elif skeleton_version == "v2":
            self._template_file_name = "network_skeleton_v2.tpl.vhd"
            if len(id) != 16:
                raise Exception(
                    f"should give an id of 16 byte. Actual length is {len(id)}"
                )
            if x_num_values > 19983:
                raise Exception(
                    "Not more than 19983 input values allowed. Actual num of inputs"
                    f" {x_num_values} ."
                )
            if y_num_values > 19983:
                raise Exception(
                    "Not more than 19983 input values allowed. Actual num of inputs"
                    f" {x_num_values} ."
                )
        else:
            raise Exception(f"Skeleton version {skeleton_version} does not exist")
        if port["x"].width > 8:
            raise Exception(
                "port x width should not be bigger than 8. You assigned "
                f" {port['x'].width=}"
            )
        if port["y"].width > 8:
            raise Exception(
                "port x width should not be bigger than 8. You assigned "
                f" {port['y'].width=}"
            )

    def save_to(self, destination: Path):
        conv = FxpConverter(FxpParams(total_bits=8, frac_bits=0, signed=False))
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=self._template_file_name,
            parameters=dict(
                name=self.name,
                network_name=self._network_name,
                data_width_in=str(self._port["x"].width),
                x_addr_width=str(self._port["x_address"].width),
                x_num_values=self._x_num_values,
                y_num_values=self._y_num_values,
                data_width_out=str(self._port["y"].width),
                y_addr_width=str(self._port["y_address"].width),
                id=", ".join(map(conv.integer_to_binary_string_vhdl, self._id)),
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


class EchoSkeletonV2:
    def __init__(self, num_values: int, bitwidth: int):
        self._num_values = num_values
        self._bitwidth = bitwidth
        self._name = "skeleton"
        if bitwidth > 8:
            raise Exception(
                "Not more than 8 bit supported by middleware. You assigned"
                f" {bitwidth} bits"
            )
        self._template_file_name = "network_skeleton_v2_echo.tpl.vhd"
        self._id = [50, 52, 48, 56, 50, 51, 69, 67, 72, 79, 83, 69, 82, 86, 69, 82]

    def save_to(self, destination: Path):
        conv = FxpConverter(FxpParams(total_bits=8, frac_bits=0, signed=False))
        template = InProjectTemplate(
            package=module_to_package(self.__module__),
            file_name=self._template_file_name,
            parameters=dict(
                name=self._name,
                data_width=str(self._bitwidth),
                num_values=str(self._num_values),
                id=", ".join(map(conv.integer_to_binary_string_vhdl, self._id)),
            ),
        )
        file = destination.as_file(".vhd")
        file.write(template)
