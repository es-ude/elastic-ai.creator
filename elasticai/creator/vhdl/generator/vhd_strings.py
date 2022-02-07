from typing import List
from os import path
from elasticai.creator.vhdl.language import CodeGenerator


def get_file_path_string(folder_names: List[str], file_name: str) -> str:
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


def get_gate_definition_string(comment: str, signal_names: List[str]) -> str:
    """
        returns the string of input gate
    Args:
        comment(str): comment for the gate definition
        signal_names (str): contains the name of signals with/without activation
    Returns:
        string of input/forget/cell/output/ gate definition or new cell state definition
    """
    signals_string = ""
    for signal in signal_names:
        signals_string += """signal {signal} : signed(DATA_WIDTH-1 downto 0):=(others=>'0');\n""".format(
            signal=signal
        )
    signals_string = comment + "\n" + signals_string + "\n\n\n"

    return signals_string


# maybe it could be more general !
def get_mac_async_architecture_behavior_string() -> CodeGenerator:
    """
    Returns:
        string of the behavior for mac_async architecture
    """
    yield "\t-- behavior: y=w1*x1+w2*x2+b"
    yield "\tproduct_1 <= shift_right((x1 * w1), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);"
    yield "\tproduct_2 <= shift_right((x2 * w2), FRAC_WIDTH)(DATA_WIDTH-1 downto 0);"
    yield "\ty <= product_1 + product_2 + b;"


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
    string = """\t{map_name}: {component_name}\n\tport map (\n""".format(
        map_name=map_name, component_name=component_name
    )
    if type(signals) == list:
        for signal in signals:
            string += """\t\t{signal},\n""".format(signal=signal)
    else:
        for signal in signals:
            string += """\t\t{val1} => {val2},\n""".format(
                val1=signal, val2=signals[signal]
            )
    # remove the last comma and new line
    string = string.removesuffix(",\n") + "\n"
    # add close bracket
    string += "\t);\n\n"
    return string


def get_define_process_string(
    process_name: str, sensitive_signals_list: List[str], behavior: str
) -> str:
    """
    Returns:
        string of lstm_process
    """
    signal_string = ""
    for signal in sensitive_signals_list:
        signal_string += signal + ","
    # remove the last comma
    signal_string = signal_string.removesuffix(",")

    return """\t{process_name}: process({signal_string})
\tbegin
\t\t{behavior}
\tend process;\n\n""".format(
        process_name=process_name, signal_string=signal_string, behavior=behavior
    )
