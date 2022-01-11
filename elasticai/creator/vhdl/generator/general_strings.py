from itertools import chain
from typing import Any, Dict

from elasticai.creator.vhdl.language import ComponentDeclaration, Entity


def get_libraries_string(math_lib: bool = False, work_lib: bool = False) -> str:
    """
    returns the string of the libraries which are imported in the vhd file
    Args:
        math_lib (bool): if True the import of the math library is added
        work_lib (bool): if True the import of the work library is added
    Returns:
        string with all library imports
    """
    lib_string = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
"""
    if math_lib or work_lib:
        if math_lib:
            lib_string = lib_string + "use ieee.math_real.all;\n"
        if work_lib:
            lib_string = lib_string + "LIBRARY work;\nuse work.all;\n"
    else:
        lib_string = lib_string + "\n"
    return lib_string + "\n"


def get_entity_or_component_string(
    entity_or_component: str,
    entity_or_component_name: str,
    data_width: int,
    frac_width: int,
    variables_dict: Dict,
    vector_len_width: int = None,
    indent: str = "",
):
    """
    returns the entity or component definition string
    Args:
        entity_or_component (str):
        entity_or_component_name (str):
        data_width (int): data width
        frac_width (int): frac width
        variables_dict (Dict): dictionary with all variables and their definition
        vector_len_width (int): default not specified, if specified added to the generic part
        indent (str): number of tabs needed, specified in string
    Returns:
        string of the entity or component definition
    """
    if entity_or_component == "entity":
        entity = Entity(f"{entity_or_component_name}")

    else:
        entity = ComponentDeclaration(f"{entity_or_component_name}")

    entity.generic_list = [
        f"DATA_WIDTH : integer := {data_width}",
        f"FRAC_WIDTH : integer := {frac_width}",
    ]
    # eventually add vector_len_width
    if vector_len_width:
        entity.generic_list.append(f"VECTOR_LEN_WIDTH : integer := {vector_len_width}")

    entity.port_list = [
        f"{variable} : {definition}" for variable, definition in variables_dict.items()
    ]
    return "\n".join(chain(entity(), [""]))


def get_signal_definitions_string(signal_dict: Dict) -> str:
    """
    returns signal definitions string
    Args:
        signal_dict (Dict): dictionary with the name of each signal and its definition
    Returns:
        string of the signal definitions
    """
    signal_dict_str = ""
    for signal in signal_dict:
        signal_dict_str = (
            signal_dict_str
            + "    signal "
            + signal
            + " : "
            + signal_dict[signal]
            + ";\n"
        )
    return signal_dict_str + "\n"


def get_architecture_header_string(architecture_name: Any, component_name: Any) -> str:
    """
    returns the architecture header string
    Args:
        architecture_name (Any): name of the architecture
        component_name (Any): name of the component
    Returns:
        string with the architecture header
    """
    return """architecture {architecture_name} of {component_name} is
\n""".format(
        architecture_name=architecture_name, component_name=component_name
    )


def get_architecture_begin_string() -> str:
    """
    returns architecture begin string
    Returns:
        string of architecture begin
    """
    return "begin\n\n"


def get_architecture_end_string(architecture_name) -> str:
    """
    returns architecture end string
    Returns:
        string of architecture end
    """
    return """end architecture {architecture_name} ; -- {architecture_name}
\n""".format(
        architecture_name=architecture_name
    )


def get_variable_definitions_string(variable_dict: Dict) -> str:
    """
    returns variable definitions string in form of variable <= variable_definition
    Args:
        variable_dict (Dict): dictionary with the name of the variable and its definition
    Returns:
        string of the variable definitions
    """
    variable_str = ""
    for variable in variable_dict:
        variable_str = (
            variable_str
            + "    {variable} <= {variable_definition};\n".format(
                variable=variable, variable_definition=variable_dict[variable]
            )
        )
    return variable_str + "\n"
