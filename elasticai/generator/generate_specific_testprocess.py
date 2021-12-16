# TODO: add more possible test process cases
from typing import List, Any


def write_function_test_process(inputs: List[Any], outputs: List[Any]) -> str:
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
            test = test + """        test_input <=  to_signed({input},16);
        wait for 1*clk_period;
        report "The value of 'test_output' is " & integer'image(to_integer(unsigned(test_output)));
        assert test_output={output} report "The test case {input} fail" severity failure;
\n""".format(input=inputs[i], output=outputs[i])
        return test
    else:
        raise TypeError(f"inputs length {len(inputs)} is different to outputs length {len(outputs)}.")
