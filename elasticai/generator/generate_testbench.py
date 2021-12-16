from typing import Any, Dict


def write_libraries(math_lib: bool = False) -> str:
    """
    returns the string of the libraries which are imported in the vhd file
    Args:
        math_lib (bool): if True the import of the math library is added
    Returns:
        string with all library imports
    """
    lib_string = """library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions
"""
    if math_lib:
        lib_string = lib_string + "use ieee.math_real.all;                 -- for the ceiling and log constant calculation function\n\n"
    else:
        lib_string = lib_string + "\n"
    return lib_string


def write_entity(entity_name: Any) -> str:
    """
    returns the entity definition string for the entity_name
    Args:
        entity_name (Any): name of the entity
    Returns:
        string with entity definition
    """
    return """entity {entity_name}_tb is
    port ( clk: out std_logic);
end entity ; -- {entity_name}_tb
\n""".format(entity_name=entity_name)


def write_architecture_header(architecture_name: Any, component_name: Any) -> str:
    """
    returns the architecture header string
    Args:
        architecture_name (Any): name of the architecture
        component_name (Any): name of the component
    Returns:
        string with the architecture header
    """
    return """architecture {architecture_name} of {component_name}_tb is
\n""".format(architecture_name=architecture_name, component_name=component_name)


def write_component(component_name: Any, data_width: Any, frac_width: Any, variables_dict: Dict[Any]) -> str:
    """
    returns the component definition string
    Args:
        component_name (Any): name of the component
        data_width (Any): data width
        frac_width (Any): frac width
        variables_dict (Dict[Any]): dictionary with all variables and the declaration if they are an input or output variable
    Returns:
        string of the component definition
    """
    variables_list_str = ""
    for variable in variables_dict:
        variables_list_str = variables_list_str + "            {variable} : {io} signed(DATA_WIDTH-1 downto 0);\n".format(variable=variable, io=variables_dict[variable])
    # remove last linebreak and semicolon
    return """    component {component_name} is
        generic (
                DATA_WIDTH : integer := {data_width};
                FRAC_WIDTH : integer := {frac_width}
            );
        port (\n""".format(component_name=component_name, data_width=data_width, frac_width=frac_width) + variables_list_str[:-2] + """
        );
    end component {component_name};
\n""".format(component_name=component_name)


def write_signal_definitions(signal_dict: Dict[Any]) -> str:
    """
    returns signal definitions string
    Args:
        signal_dict (Dict[Any]): dictionary with the name of each signal and its definition
    Returns:
        string of the signal definitions
    """
    signal_dict_str = ""
    for signal in signal_dict:
        signal_dict_str = signal_dict_str + "    signal " + signal + " : " + signal_dict[signal] + ";\n"
    return """    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------\n""" + signal_dict_str + "\n"


def write_clock_process() -> str:
    """
    returns the clock process string
    Returns:
        string of the clock process
    """
    return """begin

    clock_process : process
    begin
        clk <= '0';
        wait for clk_period/2;
        clk <= '1';
        wait for clk_period/2;
    end process; -- clock_process
\n"""


def write_uut(component_name: Any, mapping_dict: Dict[Any]) -> str:
    """
    writes the unit under test definition string
    Args:
        component_name (Any): name of the component
        mapping_dict (Dict[Any]): dictionary with the mapping of the variables to the signals
    Returns:
        string of the unit under test definition
    """
    mapping_dict_str = ""
    for mapping in mapping_dict:
        mapping_dict_str = mapping_dict_str + "    " + mapping + " => " + mapping_dict[mapping] + ",\n"
    # remove last comma and linebreak and add linebreak again
    return """    uut: {component_name}
    port map (\n""".format(component_name=component_name) + mapping_dict_str[:-2] + "\n    );\n\n"


def write_test_process_header() -> str:
    """
    returns test process header string
    Returns:
        string of test process header
    """
    return """    test_process: process is
    begin
        Report "======Simulation start======" severity Note;
\n"""


def write_test_process_end() -> str:
    """
    returns test process end string
    Returns:
        string of test process end
    """
    return """
        -- if there is no error message, that means all test case are passed.
        report "======Simulation Success======" severity Note;
        report "Please check the output message." severity Note;

        -- wait forever
        wait;

    end process; -- test_process
\n"""


def write_architecture_end(architecture_name) -> str:
    """
    returns architecture end string
    Returns:
        string of architecture end
    """
    return """end {architecture_name} ; -- {architecture_name}
\n""".format(architecture_name=architecture_name)
