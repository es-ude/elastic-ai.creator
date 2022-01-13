from elasticai.creator.vhdl.generator.generator_functions import (
    precomputed_scalar_function_process,
)
from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest


class PrecomputedScalarFunctionProcessTest(GeneratedVHDLCodeTest):
    def test_empty_x_list_y_list_one_element(self) -> None:
        code = precomputed_scalar_function_process(x_list=[], y_list=[1])
        self.assertEqual(code, 'y <= "0000000100000000";' + "\n" + "\t" + "\t")

    def test_empty_x_list_y_list_too_many_elements(self) -> None:
        self.assertRaises(
            ValueError, precomputed_scalar_function_process, x_list=[], y_list=[1, 2]
        )

    def test_empty_y_list(self) -> None:
        self.assertRaises(
            ValueError, precomputed_scalar_function_process, x_list=[], y_list=[]
        )

    def test_x_list_lengths_not_suitable_for_y_list_lengths(self) -> None:
        self.assertRaises(
            ValueError, precomputed_scalar_function_process, x_list=[1, 2], y_list=[1]
        )

    def test_x_list_with_only_one_element(self) -> None:
        expected_code = """        if int_x<256 then
                y <= "0000000100000000"; -- 256
            else
                y <= "0000001000000000"; -- 512
            end if;"""
        code = precomputed_scalar_function_process(x_list=[1], y_list=[1, 2])
        self.check_generated_code(expected_code, code)

    def test_unsorted_x_list(self) -> None:
        expected_code = """        if int_x<256 then
                    y <= "0000000100000000"; -- 256
                elsif int_x<512 then
                    y <= "0000001000000000"; -- 512
                elsif int_x<768 then
                    y <= "0000001100000000"; -- 768
                else
                    y <= "0000010000000000"; -- 1024
                end if;"""
        code = precomputed_scalar_function_process(
            x_list=[3, 1, 2], y_list=[1, 2, 3, 4]
        )
        self.check_generated_code(expected_code, code)
