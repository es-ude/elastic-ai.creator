from typing import Any, Dict, List

from elasticai.generator.generate_specific_testprocess import \
    get_test_process_for_one_input_results_in_one_output_string, \
    get_test_process_for_multiple_input_results_in_one_output_string


def get_libraries_string(math_lib: bool = False) -> str:
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


def get_entity_string(entity_name: Any, data_width, frac_width, vector_len_width=None) -> str:
    """
    returns the entity definition string for the entity_name
    Args:
        entity_name (Any): name of the entity
    Returns:
        string with entity definition
    """
    beginning_entity_str = "entity {entity_name}_tb is\n".format(entity_name=entity_name)

    generic_str = "    generic (\n"
    generic_str = generic_str + "        DATA_WIDTH : integer := {data_width};\n".format(data_width=data_width)
    generic_str = generic_str + "        FRAC_WIDTH : integer := {frac_width};\n".format(frac_width=frac_width)
    if vector_len_width:
        generic_str = generic_str + "        VECTOR_LEN_WIDTH : integer := {vector_len_width};\n".format(
            vector_len_width=vector_len_width)
    beginning_entity_str = beginning_entity_str + generic_str[:-2] + """
        );\n"""

    return beginning_entity_str + """    port ( clk: out std_logic);
end entity ; -- {entity_name}_tb
\n""".format(entity_name=entity_name)


def get_architecture_header_string(architecture_name: Any, component_name: Any) -> str:
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


def get_component_string(component_name: Any, data_width: Any, frac_width: Any, variables_dict: Dict, vector_len_width=None) -> str:
    """
    returns the component definition string
    Args:
        component_name (Any): name of the component
        data_width (Any): data width
        frac_width (Any): frac width
        variables_dict (Dict): dictionary with all variables and their definition
        vector_len_width (): default not specified, if specified added to the generic part
    Returns:
        string of the component definition
    """
    first_part_of_string = """    ------------------------------------------------------------
    -- Declare Components for testing
    ------------------------------------------------------------
    component {component_name} is
        generic (
                DATA_WIDTH : integer := {data_width};
                FRAC_WIDTH : integer := {frac_width}""".format(component_name=component_name, data_width=data_width, frac_width=frac_width)

    # eventually add vector_len_width
    if vector_len_width:
        second_part_of_string = f";\n                VECTOR_LEN_WIDTH : integer := {vector_len_width}\n".format(vector_len_width=vector_len_width)
    else:
        second_part_of_string = "\n"

    closing_of_generic_string = """            );
        port (\n"""

    # add variables
    variables_list_str = ""
    for variable in variables_dict:
        variables_list_str = variables_list_str + "            {variable} : {definition};\n".format(variable=variable, definition=variables_dict[variable])
    # remove last linebreak and semicolon
    return first_part_of_string + second_part_of_string + closing_of_generic_string + variables_list_str[:-2] + """
        );
    end component {component_name};
\n""".format(component_name=component_name)


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
        signal_dict_str = signal_dict_str + "    signal " + signal + " : " + signal_dict[signal] + ";\n"
    return """    ------------------------------------------------------------
    -- Testbench Internal Signals
    ------------------------------------------------------------\n""" + signal_dict_str + "\n"


def get_type_definitions_string(type_dict: Dict) -> str:
    """
    returns types definitions string
    Args:
        type_dict (Dict): dictionary with the name of each type and its definition
    Returns:
        string of the type definitions
    """
    type_dict_str = """    ------------------------------------------------------------
    -- Testbench Data Type
    ------------------------------------------------------------\n"""
    for type_variable in type_dict:
        type_dict_str = type_dict_str + "    type {type_variable} is {type_definition};\n".\
            format(type_variable=type_variable, type_definition=type_dict[type_variable])
    return type_dict_str + "\n"


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
        variable_str = variable_str + "    {variable} <= {variable_definition};\n".\
            format(variable=variable, variable_definition=variable_dict[variable])
    return variable_str + "\n"


def get_begin_architecture_string() -> str:
    """
    returns architecture begin string
    Returns:
        string of architecture end
    """
    return "begin\n\n"


def get_clock_process_string(clock_name="clk") -> str:
    """
    returns the clock process string
    Returns:
        string of the clock process
    """
    return """    clock_process : process
    begin
        {clock_name} <= '0';
        wait for clk_period/2;
        {clock_name} <= '1';
        wait for clk_period/2;
    end process; -- clock_process
\n""".format(clock_name=clock_name)


def get_uut_string(component_name: Any, mapping_dict: Dict) -> str:
    """
    writes the unit under test definition string
    Args:
        component_name (Any): name of the component
        mapping_dict (Dict): dictionary with the mapping of the variables to the signals
    Returns:
        string of the unit under test definition
    """
    mapping_dict_str = ""
    for mapping in mapping_dict:
        mapping_dict_str = mapping_dict_str + "        " + mapping + " => " + mapping_dict[mapping] + ",\n"
    # remove last comma and linebreak and add linebreak again
    return """    uut: {component_name}
    port map (\n""".format(component_name=component_name) + mapping_dict_str[:-2] + "\n    );\n\n"


def get_test_process_header_string() -> str:
    """
    returns test process header string
    Returns:
        string of test process header
    """
    return """    test_process: process is
    begin
        Report "======Simulation start======" severity Note;
\n"""


def get_test_process_end_string() -> str:
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


def get_architecture_end_string(architecture_name) -> str:
    """
    returns architecture end string
    Returns:
        string of architecture end
    """
    return """end {architecture_name} ; -- {architecture_name}
\n""".format(architecture_name=architecture_name)


def write_testbench_file(
        path_to_testbench: str,
        test_bench_file_name: str,
        component_name: str,
        architecture_name: str,
        data_width: int,
        frac_width: int,
        component_variables_dict: Dict,
        signal_definitions_dict: Dict,
        uut_mapping_dict: Dict,
        inputs_for_testcases: List,
        outputs_for_testcases: List,
        output_name_for_testcases: str,
        input_name_for_testcases: str = None,
        math_lib: bool = False,
        vector_len_width: int = None,
        type_definitions_dict: Dict = None,
        clock_name: str = "clk",
        variable_definitions_before_test_process_dict: Dict = None,
        variable_definitions_in_test_process_dict: Dict = None
) -> None:
    """
    writes the testbench file for the specifications
    Args:
        path_to_testbench (str): path where testbench is located
        test_bench_file_name (str): name of the testbench file name
        component_name (str): component name
        architecture_name (str): architecture name
        data_width (int): data width
        frac_width (int): frac width
        component_variables_dict (Dict): variable definition of the components
        signal_definitions_dict (Dict): signal definition
        uut_mapping_dict (Dict): mapping in the uut
        inputs_for_testcases (List): List of inputs
        outputs_for_testcases (List): List of corresponding outputs for the inputs
        output_name_for_testcases (str): name of the output variable
        input_name_for_testcases (str): name of the input variable, default None (this determines if we have multiple inputs or only one)
        math_lib (bool): True, if the math library should be imported in the testbench file, default False
        vector_len_width (int): vector length width, not always defined, default None
        type_definitions_dict (Dict): possible definitions of new types, default None
        clock_name (str): name of the clock, default "clk"
        variable_definitions_before_test_process_dict (Dict): possible definitions of variables before the test process, default None
        variable_definitions_in_test_process_dict (Dict): possible definitions of variables in the test process, default None
    """
    with open(path_to_testbench + test_bench_file_name, 'w') as f:
        f.write(get_libraries_string(math_lib=math_lib))
        f.write(get_entity_string(entity_name=component_name, data_width=data_width, frac_width=frac_width,
                                  vector_len_width=vector_len_width))
        f.write(get_architecture_header_string(architecture_name=architecture_name, component_name=component_name))
        if type_definitions_dict:
            f.write(get_type_definitions_string(type_dict=type_definitions_dict))
        f.write(get_signal_definitions_string(signal_dict=signal_definitions_dict))
        f.write(get_component_string(component_name=component_name, data_width=data_width, frac_width=frac_width,
                                     vector_len_width=vector_len_width, variables_dict=component_variables_dict))
        f.write(get_begin_architecture_string())
        f.write(get_clock_process_string(clock_name=clock_name))
        f.write(get_uut_string(component_name=component_name, mapping_dict=uut_mapping_dict))
        if variable_definitions_before_test_process_dict:
            f.write(get_variable_definitions_string(variable_dict=variable_definitions_before_test_process_dict))
        f.write(get_test_process_header_string())
        if variable_definitions_in_test_process_dict:
            f.write(get_variable_definitions_string(variable_dict=variable_definitions_in_test_process_dict))
        # when one input results in one output
        if input_name_for_testcases:
            f.write(get_test_process_for_one_input_results_in_one_output_string(
                inputs=inputs_for_testcases,
                outputs=outputs_for_testcases,
                input_name=input_name_for_testcases,
                output_name=output_name_for_testcases))
        else:
            f.write(get_test_process_for_multiple_input_results_in_one_output_string(
                inputs=inputs_for_testcases,
                outputs=outputs_for_testcases,
                output_name=output_name_for_testcases))
        f.write(get_test_process_end_string())
        f.write(get_architecture_end_string(architecture_name=architecture_name))
