from typing import Any


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
    if math_lib:
        lib_string = lib_string + "use ieee.math_real.all;\n"
    if work_lib:
        lib_string = lib_string + "LIBRARY work;\nuse work.all;\n"
    else:
        lib_string = lib_string + "\n"
    return lib_string + '\n'


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
\n""".format(architecture_name=architecture_name, component_name=component_name)


def get_begin_architecture_string() -> str:
    """
    returns architecture begin string
    Returns:
        string of architecture end
    """
    return "begin\n\n"


def get_architecture_end_string(architecture_name) -> str:
    """
    returns architecture end string
    Returns:
        string of architecture end
    """
    return """end {architecture_name} ; -- {architecture_name}
\n""".format(architecture_name=architecture_name)
