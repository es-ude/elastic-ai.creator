from typing import List, Any, Dict


def get_test_process_for_one_input_results_in_one_output_string(
    inputs: List[Any], outputs: List[Any], input_name: str, output_name: str
) -> str:
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
\n""".format(
                input=inputs[i],
                output=outputs[i],
                input_name=input_name,
                output_name=output_name,
            )
        return test
    else:
        raise TypeError(
            f"inputs length {len(inputs)} is different to outputs length {len(outputs)}."
        )


def get_test_process_for_multiple_input_results_in_one_output_string(
    inputs: List[Dict], outputs: List[Any], output_name: str
) -> str:
    """
    returns test process cases for multiple inputs which result in one output
    Args:
        inputs (List[Dict]): list of inputs for one output, the list includes a dictionary which defines each input
        outputs (List): output for each test case
        output_name (str): name of the output variable
    Returns:
        string of the testcases of each input dictionary and output pair
    """
    test = ""
    if len(inputs) == len(outputs):
        for i in range(len(outputs)):
            input_dict = inputs[i]
            for input in input_dict:
                test = test + "        {input_name} <= {input};\n".format(
                    input_name=input, input=input_dict[input]
                )
            test = (
                test
                + """
        reset <= '1';
        wait for 2*clk_period;
        wait until clock = '0';
        reset <= '0';
        wait until ready = '1';

        report "expected output is {output}, value of '{output_name}' is " & integer'image(to_integer(signed({output_name})));
        assert {output_name} = {output} report "The {counter}. test case fail" severity error;
        reset <= '1';
        wait for 1*clk_period;
\n""".format(
                    output=outputs[i], output_name=output_name, counter=i
                )
            )
        return test
    else:
        raise TypeError(
            f"inputs length {len(inputs)} is different to outputs length {len(outputs)}."
        )
