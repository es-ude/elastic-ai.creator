import warnings
from functools import partial
from pathlib import Path

from elasticai.creator.file_generation.template import InProjectTemplate, save_template
from elasticai.creator.vhdl.code_generation.code_abstractions import (
    to_vhdl_binary_string,
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
                    f"Skeleton V1 might be deprecated in the future. Consider using"
                    f" Skeleton V2 instead."
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
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
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
                id=", ".join(
                    map(partial(to_vhdl_binary_string, number_of_bits=8), self._id)
                ),
            ),
        )
        save_template(template, destination.with_suffix(".vhd"))


class LSTMSkeleton:
    def __init__(self, network_name: str):
        self.name = "skeleton"
        self._network_name = network_name

    def save_to(self, destination: Path):
        template = InProjectTemplate(
            package=InProjectTemplate.module_to_package(self.__module__),
            file_name="lstm_network_skeleton.tpl.vhd",
            parameters=dict(name=self.name, network_name=self._network_name),
        )
        save_template(template, destination.with_suffix(".vhd"))
