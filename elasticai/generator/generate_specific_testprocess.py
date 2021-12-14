# TODO: add more possible test process cases

def write_function_test_process(inputs, outputs):
    """
    writes test process cases for a function like sigmoid or tanh
    Args:
        inputs ():
        outputs ():

    Returns:

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