import numpy as np

from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.generator.precomputed_scalar_function import Sigmoid


class SigmoidTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        expected_code = """-- A LUT version of sigmoid
        library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;

        entity sigmoid is
        	generic (
        			DATA_WIDTH : integer := 16;
        			FRAC_WIDTH : integer := 8
        		);
            port (
                x : in signed(DATA_WIDTH-1 downto 0);
                y : out signed(DATA_WIDTH-1 downto 0)
            );

        end entity sigmoid;

        architecture rtl of sigmoid is
        begin

            sigmoid_process: process(x)
            variable int_x: integer := 0;
            begin
                int_x := to_integer(x);

                if int_x<-1280 then
                    y <= "0000000000000000"; -- 0
                elsif int_x<-1240 then
                    y <= "0000000000000001"; -- 1
                elsif int_x<-1201 then
                    y <= "0000000000000001"; -- 1
                elsif int_x<-1161 then
                    y <= "0000000000000010"; -- 2
                elsif int_x<-1122 then
                    y <= "0000000000000010"; -- 2
                elsif int_x<-1083 then
                    y <= "0000000000000011"; -- 3
                elsif int_x<-1043 then
                    y <= "0000000000000011"; -- 3
                elsif int_x<-1004 then
                    y <= "0000000000000100"; -- 4
                elsif int_x<-964 then
                    y <= "0000000000000100"; -- 4
                elsif int_x<-925 then
                    y <= "0000000000000101"; -- 5
                elsif int_x<-886 then
                    y <= "0000000000000110"; -- 6
                elsif int_x<-846 then
                    y <= "0000000000000111"; -- 7
                elsif int_x<-807 then
                    y <= "0000000000001001"; -- 9
                elsif int_x<-768 then
                    y <= "0000000000001010"; -- 10
                elsif int_x<-728 then
                    y <= "0000000000001100"; -- 12
                elsif int_x<-689 then
                    y <= "0000000000001110"; -- 14
                elsif int_x<-649 then
                    y <= "0000000000010000"; -- 16
                elsif int_x<-610 then
                    y <= "0000000000010010"; -- 18
                elsif int_x<-571 then
                    y <= "0000000000010101"; -- 21
                elsif int_x<-531 then
                    y <= "0000000000011000"; -- 24
                elsif int_x<-492 then
                    y <= "0000000000011100"; -- 28
                elsif int_x<-452 then
                    y <= "0000000000100000"; -- 32
                elsif int_x<-413 then
                    y <= "0000000000100101"; -- 37
                elsif int_x<-374 then
                    y <= "0000000000101010"; -- 42
                elsif int_x<-334 then
                    y <= "0000000000110000"; -- 48
                elsif int_x<-295 then
                    y <= "0000000000110110"; -- 54
                elsif int_x<-256 then
                    y <= "0000000000111101"; -- 61
                elsif int_x<-216 then
                    y <= "0000000001000100"; -- 68
                elsif int_x<-177 then
                    y <= "0000000001001100"; -- 76
                elsif int_x<-137 then
                    y <= "0000000001010101"; -- 85
                elsif int_x<-98 then
                    y <= "0000000001011110"; -- 94
                elsif int_x<-59 then
                    y <= "0000000001100111"; -- 103
                elsif int_x<-19 then
                    y <= "0000000001110001"; -- 113
                elsif int_x<19 then
                    y <= "0000000001111011"; -- 123
                elsif int_x<59 then
                    y <= "0000000010000100"; -- 132
                elsif int_x<98 then
                    y <= "0000000010001110"; -- 142
                elsif int_x<137 then
                    y <= "0000000010011000"; -- 152
                elsif int_x<177 then
                    y <= "0000000010100001"; -- 161
                elsif int_x<216 then
                    y <= "0000000010101010"; -- 170
                elsif int_x<256 then
                    y <= "0000000010110011"; -- 179
                elsif int_x<295 then
                    y <= "0000000010111011"; -- 187
                elsif int_x<334 then
                    y <= "0000000011000010"; -- 194
                elsif int_x<374 then
                    y <= "0000000011001001"; -- 201
                elsif int_x<413 then
                    y <= "0000000011001111"; -- 207
                elsif int_x<452 then
                    y <= "0000000011010101"; -- 213
                elsif int_x<492 then
                    y <= "0000000011011010"; -- 218
                elsif int_x<531 then
                    y <= "0000000011011111"; -- 223
                elsif int_x<571 then
                    y <= "0000000011100011"; -- 227
                elsif int_x<610 then
                    y <= "0000000011100111"; -- 231
                elsif int_x<649 then
                    y <= "0000000011101010"; -- 234
                elsif int_x<689 then
                    y <= "0000000011101101"; -- 237
                elsif int_x<728 then
                    y <= "0000000011101111"; -- 239
                elsif int_x<768 then
                    y <= "0000000011110001"; -- 241
                elsif int_x<807 then
                    y <= "0000000011110011"; -- 243
                elsif int_x<846 then
                    y <= "0000000011110101"; -- 245
                elsif int_x<886 then
                    y <= "0000000011110110"; -- 246
                elsif int_x<925 then
                    y <= "0000000011111000"; -- 248
                elsif int_x<964 then
                    y <= "0000000011111001"; -- 249
                elsif int_x<1004 then
                    y <= "0000000011111010"; -- 250
                elsif int_x<1043 then
                    y <= "0000000011111011"; -- 251
                elsif int_x<1083 then
                    y <= "0000000011111011"; -- 251
                elsif int_x<1122 then
                    y <= "0000000011111100"; -- 252
                elsif int_x<1161 then
                    y <= "0000000011111100"; -- 252
                elsif int_x<1201 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<1240 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<1280 then
                    y <= "0000000011111110"; -- 254
                else
                    y <= "0000000100000000"; -- 256
                end if;

            end process sigmoid_process;
        end architecture rtl;
        """
        sigmoid = Sigmoid(
            data_width=16,
            frac_width=8,
            x=np.linspace(-5, 5, 66),
            component_name="sigmoid",
        )
        sigmoid_code = sigmoid()
        sigmoid_code_str = ""
        for line in sigmoid_code:
            sigmoid_code_str += line + "\n"
        self.check_generated_code(expected_code, sigmoid_code_str)
