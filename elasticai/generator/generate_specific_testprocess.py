# TODO: add more possible test process cases
from typing import List, Any, Dict


def write_function_test_process_for_one_input_results_in_one_output(inputs: List[Any], outputs: List[Any], input_name, output_name) -> str:
    """
    writes test process cases for a function like sigmoid or tanh
    Args:
        inputs (List[Any]): list of the inputs
        outputs (List[Outputs]): list of outputs
    Returns:
        string of the testcases of each input and output pair
    """

    test = ""
    if len(inputs) == len(outputs):
        for i in range(len(inputs)):
            test = test + """        {input_name} <=  to_signed({input},16);
        wait for 1*clk_period;
        report "The value of '{output_name}' is " & integer'image(to_integer(unsigned({output_name})));
        assert {output_name}={output} report "The test case {input} fail" severity failure;
\n""".format(input=inputs[i], output=outputs[i], input_name=input_name, output_name=output_name)
        return test
    else:
        raise TypeError(f"inputs length {len(inputs)} is different to outputs length {len(outputs)}.")


def write_function_test_process_for_multiple_input_results_in_one_output(inputs_with_names: Dict, outputs: List[Any], output_name) -> str:
    test = ""
    for i in range(len(outputs)):
        for inputs in inputs_with_names:
            test = test + "        {input_name} <=  to_signed({input},16);\n".format(input_name=inputs, input=inputs_with_names[inputs])
        test = test + """        wait for 1*clk_period;
        report "The value of '{output_name}' is " & integer'image(to_integer(unsigned({output_name})));
        assert {output_name} = {output} report "The {counter}. test case fail" severity error;
\n""".format(output=outputs[i], output_name=output_name, counter=i+1)
        return test
