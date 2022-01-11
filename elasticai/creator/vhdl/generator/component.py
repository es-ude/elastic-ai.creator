from abc import ABC, abstractmethod
from functools import partial
from itertools import chain

import torch.nn

from elasticai.creator.vhdl.generator.general_strings import (
    get_libraries_string,
    get_entity_or_component_string,
    get_architecture_header_string,
    get_architecture_end_string,
)
from elasticai.creator.vhdl.generator.generator_functions import (
    sigmoid_process,
    tanh_process,
)
from elasticai.creator.vhdl.generator.vhd_strings import get_process_string
from elasticai.creator.vhdl.language import (
    Entity,
    InterfaceVariable,
    DataType,
)


class Component(ABC):
    def __init__(self, data_width, frac_width, x, component_name=None):
        self.component_name = self._get_lower_case_class_name_or_component_name(
            component_name=component_name
        )
        self.data_width = data_width
        self.frac_width = frac_width
        self.x = x

    @classmethod
    def _get_lower_case_class_name_or_component_name(cls, component_name):
        if component_name is None:
            return cls.__name__.lower()
        return component_name

    @property
    def file_name(self) -> str:
        return f"{self.component_name}.vhd"

    @property
    def architecture_name(self) -> str:
        return f"{self.component_name}_rtl"

    @abstractmethod
    def build(self) -> str:
        pass


class DataWidthVariable(InterfaceVariable):
    def __init__(self, value: int):
        super().__init__(
            identifier="DATA_WIDTH", variable_type=DataType.INTEGER, value=f"{value}"
        )


class FracWidthVariable(InterfaceVariable):
    def __init__(self, value: int):
        super().__init__(
            identifier="FRAC_WIDTH", variable_type=DataType.INTEGER, value=f"{value}"
        )


class Sigmoid(Component):
    def build(self) -> str:
        entity = Entity(self.component_name)
        entity.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
        ]
        entity.port_list = [
            "x : in signed(DATA_WIDTH-1 downto 0)",
            "y : out signed(DATA_WIDTH-1 downto 0)",
        ]
        code = "\n".join(chain([get_libraries_string()], chain(entity()), [""]))
        code += get_architecture_header_string(
            architecture_name=self.architecture_name, component_name=self.component_name
        )
        code += get_process_string(
            component_name=self.component_name,
            lookup_table_generator_function=sigmoid_process(
                x_list=torch.as_tensor(self.x), function=torch.nn.Sigmoid()
            ),
        )
        code += get_architecture_end_string(architecture_name=self.architecture_name)
        return code


class Tanh(Component):
    def build(self) -> str:
        code = ""
        entity = Entity(self.component_name)
        entity.generic_list = [
            f"DATA_WIDTH : integer := {self.data_width}",
            f"FRAC_WIDTH : integer := {self.frac_width}",
        ]
        entity.port_list = [
            "x : in signed(DATA_WIDTH-1 downto 0)",
            "y : out signed(DATA_WIDTH-1 downto 0)",
        ]
        string_builders = [
            get_libraries_string,
            lambda: "\n".join(chain(entity(), [""])),
            partial(
                get_architecture_header_string,
                architecture_name=self.architecture_name,
                component_name=self.component_name,
            ),
            partial(
                get_process_string,
                component_name=self.component_name,
                lookup_table_generator_function=tanh_process(self.x),
            ),
            partial(
                get_architecture_end_string, architecture_name=self.architecture_name
            ),
        ]
        for function in string_builders:
            code += function()
        return code
