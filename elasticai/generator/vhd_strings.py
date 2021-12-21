from typing import Any, Dict, List, Iterable
from os import path


def get_process_string(component_name: str, process_name) -> str:
    """
    returns the string of a process Block
    Args:
        component_name (str): the name of the component
        process_name (str): a function in generator_functions which generate look up table for sigmoid/tanh
    Returns:
        string as vhdl process Block
    """
    return """begin 
    {component_name}_process:process(x)
    variable int_x: integer := 0;
    begin
        int_x := to_integer(x);
        
        {generate_process}
    end process;                
    \n""".format(component_name=component_name, generate_process=process_name)


def get_file_path(folder_names: List[str], file_name: str) -> str:
    """
    returns String of a file path
    Args:
        folder_names (List(str)): the name of the folders
        file_name (str): the name of the file
    Returns:
        string of the full path to a given filename
    """
    base_path = path.dirname(__file__)
    file_path = base_path
    for folder_name in folder_names:
        file_path = path.abspath(path.join(file_path, folder_name))
    file_path = path.abspath(path.join(file_path, file_name))
    return file_path


def get_input_gate_string(without_activation: str, with_activation: str) -> str:
    """
        returns the string of input gate
    Args:
        without_activation (str): name of a signal
        with_activation (str): name of a signal
    Returns:
        string of input gate definition
    """
    string = """-- Intermediate results
-- Input gate without/with activation
-- i = \sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
        signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""").format(
        without_activation=without_activation, with_activation=with_activation)
    return string


def get_forget_gate_string(without_activation: str, with_activation: str) -> str:
    """
        returns the string of forget gate definitions
    Args:
        without_activation (str): name of a signal
        with_activation (str): name of a signal
    Returns:
        string of forget gate definition
    """
    string = """-- Forget gate without/with activation
-- f = \sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""").format(
        without_activation=without_activation, with_activation=with_activation)
    return string


def get_cell_gate_string(without_activation: str, with_activation: str) -> str:
    """
        returns the string of cell gate definitions
    Args:
        without_activation (str): name of a signal
        with_activation (str): name of a signal
    Returns:
        string of cell gate definition
    """
    string = """-- Cell gate without/with activation
-- g = \\tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""".format(
            without_activation=without_activation, with_activation=with_activation))
    return string


def get_output_gate_string(without_activation: str, with_activation: str) -> str:
    """
        returns the string of output gate definitions
    Args:
        without_activation (str): name of a signal
        with_activation (str): name of a signal
    Returns:
        string of output gate definition
    """
    string = """-- Output gate without/with activation
-- o = \sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho})\n""" + (
        """signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
        signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n""".format(
            without_activation=without_activation, with_activation=with_activation))
    return string


def get_new_cell_state_string(without_activation: str, with_activation: str) -> str:
    """
        returns the string of new cell state definitions
    Args:
        without_activation (str): name of a signal
        with_activation (str): name of a signal
    Returns:
        string of output gate definition
    """
    return """-- new_cell_state without/with activation
-- c' = f * c + i * g 
signal {without_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');
signal {with_activation} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');

-- h' = o * \\tanh(c')
signal h_new : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n\n\n""".format(without_activation=without_activation,
                                                                             with_activation=with_activation)


# maybe it could be more general !
def get_mac_async_architecture_behavior_string() -> str:
    """
    Returns:
        string of the behavior for mac_async architecture
    """
    return """    -- behavior: y=w1*x1+w2*x2+b
    product_1 <= shift_right((x1 * w1), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    product_2 <= shift_right((x2 * w2), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    y <= product_1 + product_2 + b;\n"""


def get_port_map_string(map_name: str, component_name: str, signals) -> str:
    """
    returns port map string
    Args:
        map_name (str): the name of the map
        component_name(str): the name of the component
        signals(Dict/List): dictionary with the name of each signal and its definition / List of each signal
    Returns:
        string of port map
    """
    string = """    {map_name}: {component_name}\n    port map (\n""".format(map_name=map_name,
                                                                             component_name=component_name)
    if type(signals) == list:
        for signal in signals:
            string += """        {signal},\n""".format(signal=signal)
    else:
        for signal in signals:
            string += """        {val1} => {val2},\n""".format(val1=signal, val2=signals[signal])
    # remove the last comma and new line
    string = string.removesuffix(",\n") + "\n"
    # add close bracket
    string += "    );\n\n"
    return string


# maybe it could be more general !
def get_lstm_process_string() -> str:
    """
    Returns:
        string of lstm_process
    """
    return """    H_OUT_PROCESS: process(o,c_new)
    begin
        h_new <= shift_right((o*c_new), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);
    end process;\n\n"""
