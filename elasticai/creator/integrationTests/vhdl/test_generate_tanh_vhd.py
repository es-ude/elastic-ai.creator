import numpy as np

from elasticai.creator.tests.vhdl.vhdl_file_testcase import GeneratedVHDLCodeTest
from elasticai.creator.vhdl.number_representations import float_values_to_fixed_point
from elasticai.creator.vhdl.precomputed_scalar_function import Tanh


class GenerateTanhVhdTest(GeneratedVHDLCodeTest):
    def test_compare_files(self) -> None:
        expected_code = """library ieee;
        use ieee.std_logic_1164.all;
        use ieee.numeric_std.all;


        entity tanh is
            generic (
                DATA_WIDTH : integer := 16;
                FRAC_WIDTH : integer := 8
            );
            port (
                x : in signed(DATA_WIDTH-1 downto 0);
                y : out signed(DATA_WIDTH-1 downto 0)
            );
        end entity tanh;

        architecture rtl of tanh is

        begin
            tanh_process: process(x)
            variable int_x: integer := 0;
            begin
                int_x := to_integer(x);

                if int_x<-1280 then
                    y <= "1111111100000000"; -- -256
                elsif int_x<-1270 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1260 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1250 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1240 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1230 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1220 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1210 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1200 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1190 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1180 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1170 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1160 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1151 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1141 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1131 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1121 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1111 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1101 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1091 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1081 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1071 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1061 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1051 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1041 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1031 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1022 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1012 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-1002 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-992 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-982 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-972 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-962 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-952 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-942 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-932 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-922 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-912 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-902 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-893 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-883 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-873 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-863 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-853 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-843 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-833 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-823 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-813 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-803 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-793 then
                    y <= "1111111100000001"; -- -255
                elsif int_x<-783 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-773 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-764 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-754 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-744 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-734 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-724 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-714 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-704 then
                    y <= "1111111100000010"; -- -254
                elsif int_x<-694 then
                    y <= "1111111100000011"; -- -253
                elsif int_x<-684 then
                    y <= "1111111100000011"; -- -253
                elsif int_x<-674 then
                    y <= "1111111100000011"; -- -253
                elsif int_x<-664 then
                    y <= "1111111100000011"; -- -253
                elsif int_x<-654 then
                    y <= "1111111100000011"; -- -253
                elsif int_x<-644 then
                    y <= "1111111100000100"; -- -252
                elsif int_x<-635 then
                    y <= "1111111100000100"; -- -252
                elsif int_x<-625 then
                    y <= "1111111100000100"; -- -252
                elsif int_x<-615 then
                    y <= "1111111100000100"; -- -252
                elsif int_x<-605 then
                    y <= "1111111100000101"; -- -251
                elsif int_x<-595 then
                    y <= "1111111100000101"; -- -251
                elsif int_x<-585 then
                    y <= "1111111100000101"; -- -251
                elsif int_x<-575 then
                    y <= "1111111100000110"; -- -250
                elsif int_x<-565 then
                    y <= "1111111100000110"; -- -250
                elsif int_x<-555 then
                    y <= "1111111100000111"; -- -249
                elsif int_x<-545 then
                    y <= "1111111100000111"; -- -249
                elsif int_x<-535 then
                    y <= "1111111100001000"; -- -248
                elsif int_x<-525 then
                    y <= "1111111100001000"; -- -248
                elsif int_x<-515 then
                    y <= "1111111100001001"; -- -247
                elsif int_x<-506 then
                    y <= "1111111100001001"; -- -247
                elsif int_x<-496 then
                    y <= "1111111100001010"; -- -246
                elsif int_x<-486 then
                    y <= "1111111100001011"; -- -245
                elsif int_x<-476 then
                    y <= "1111111100001100"; -- -244
                elsif int_x<-466 then
                    y <= "1111111100001101"; -- -243
                elsif int_x<-456 then
                    y <= "1111111100001110"; -- -242
                elsif int_x<-446 then
                    y <= "1111111100001111"; -- -241
                elsif int_x<-436 then
                    y <= "1111111100010000"; -- -240
                elsif int_x<-426 then
                    y <= "1111111100010001"; -- -239
                elsif int_x<-416 then
                    y <= "1111111100010010"; -- -238
                elsif int_x<-406 then
                    y <= "1111111100010100"; -- -236
                elsif int_x<-396 then
                    y <= "1111111100010101"; -- -235
                elsif int_x<-386 then
                    y <= "1111111100010111"; -- -233
                elsif int_x<-377 then
                    y <= "1111111100011000"; -- -232
                elsif int_x<-367 then
                    y <= "1111111100011010"; -- -230
                elsif int_x<-357 then
                    y <= "1111111100011100"; -- -228
                elsif int_x<-347 then
                    y <= "1111111100011110"; -- -226
                elsif int_x<-337 then
                    y <= "1111111100100000"; -- -224
                elsif int_x<-327 then
                    y <= "1111111100100011"; -- -221
                elsif int_x<-317 then
                    y <= "1111111100100101"; -- -219
                elsif int_x<-307 then
                    y <= "1111111100101000"; -- -216
                elsif int_x<-297 then
                    y <= "1111111100101011"; -- -213
                elsif int_x<-287 then
                    y <= "1111111100101110"; -- -210
                elsif int_x<-277 then
                    y <= "1111111100110010"; -- -206
                elsif int_x<-267 then
                    y <= "1111111100110101"; -- -203
                elsif int_x<-257 then
                    y <= "1111111100111001"; -- -199
                elsif int_x<-248 then
                    y <= "1111111100111101"; -- -195
                elsif int_x<-238 then
                    y <= "1111111101000001"; -- -191
                elsif int_x<-228 then
                    y <= "1111111101000110"; -- -186
                elsif int_x<-218 then
                    y <= "1111111101001010"; -- -182
                elsif int_x<-208 then
                    y <= "1111111101001111"; -- -177
                elsif int_x<-198 then
                    y <= "1111111101010101"; -- -171
                elsif int_x<-188 then
                    y <= "1111111101011010"; -- -166
                elsif int_x<-178 then
                    y <= "1111111101100000"; -- -160
                elsif int_x<-168 then
                    y <= "1111111101100111"; -- -153
                elsif int_x<-158 then
                    y <= "1111111101101101"; -- -147
                elsif int_x<-148 then
                    y <= "1111111101110100"; -- -140
                elsif int_x<-138 then
                    y <= "1111111101111011"; -- -133
                elsif int_x<-128 then
                    y <= "1111111110000010"; -- -126
                elsif int_x<-119 then
                    y <= "1111111110001010"; -- -118
                elsif int_x<-109 then
                    y <= "1111111110010001"; -- -111
                elsif int_x<-99 then
                    y <= "1111111110011010"; -- -102
                elsif int_x<-89 then
                    y <= "1111111110100010"; -- -94
                elsif int_x<-79 then
                    y <= "1111111110101011"; -- -85
                elsif int_x<-69 then
                    y <= "1111111110110100"; -- -76
                elsif int_x<-59 then
                    y <= "1111111110111101"; -- -67
                elsif int_x<-49 then
                    y <= "1111111111000111"; -- -57
                elsif int_x<-39 then
                    y <= "1111111111010000"; -- -48
                elsif int_x<-29 then
                    y <= "1111111111011010"; -- -38
                elsif int_x<-19 then
                    y <= "1111111111100100"; -- -28
                elsif int_x<-9 then
                    y <= "1111111111101110"; -- -18
                elsif int_x<0 then
                    y <= "1111111111111000"; -- -8
                elsif int_x<9 then
                    y <= "0000000000000000"; -- 0
                elsif int_x<19 then
                    y <= "0000000000001000"; -- 8
                elsif int_x<29 then
                    y <= "0000000000010010"; -- 18
                elsif int_x<39 then
                    y <= "0000000000011100"; -- 28
                elsif int_x<49 then
                    y <= "0000000000100110"; -- 38
                elsif int_x<59 then
                    y <= "0000000000110000"; -- 48
                elsif int_x<69 then
                    y <= "0000000000111001"; -- 57
                elsif int_x<79 then
                    y <= "0000000001000011"; -- 67
                elsif int_x<89 then
                    y <= "0000000001001100"; -- 76
                elsif int_x<99 then
                    y <= "0000000001010101"; -- 85
                elsif int_x<109 then
                    y <= "0000000001011110"; -- 94
                elsif int_x<119 then
                    y <= "0000000001100110"; -- 102
                elsif int_x<128 then
                    y <= "0000000001101111"; -- 111
                elsif int_x<138 then
                    y <= "0000000001110110"; -- 118
                elsif int_x<148 then
                    y <= "0000000001111110"; -- 126
                elsif int_x<158 then
                    y <= "0000000010000101"; -- 133
                elsif int_x<168 then
                    y <= "0000000010001100"; -- 140
                elsif int_x<178 then
                    y <= "0000000010010011"; -- 147
                elsif int_x<188 then
                    y <= "0000000010011001"; -- 153
                elsif int_x<198 then
                    y <= "0000000010100000"; -- 160
                elsif int_x<208 then
                    y <= "0000000010100110"; -- 166
                elsif int_x<218 then
                    y <= "0000000010101011"; -- 171
                elsif int_x<228 then
                    y <= "0000000010110001"; -- 177
                elsif int_x<238 then
                    y <= "0000000010110110"; -- 182
                elsif int_x<248 then
                    y <= "0000000010111010"; -- 186
                elsif int_x<257 then
                    y <= "0000000010111111"; -- 191
                elsif int_x<267 then
                    y <= "0000000011000011"; -- 195
                elsif int_x<277 then
                    y <= "0000000011000111"; -- 199
                elsif int_x<287 then
                    y <= "0000000011001011"; -- 203
                elsif int_x<297 then
                    y <= "0000000011001110"; -- 206
                elsif int_x<307 then
                    y <= "0000000011010010"; -- 210
                elsif int_x<317 then
                    y <= "0000000011010101"; -- 213
                elsif int_x<327 then
                    y <= "0000000011011000"; -- 216
                elsif int_x<337 then
                    y <= "0000000011011011"; -- 219
                elsif int_x<347 then
                    y <= "0000000011011101"; -- 221
                elsif int_x<357 then
                    y <= "0000000011100000"; -- 224
                elsif int_x<367 then
                    y <= "0000000011100010"; -- 226
                elsif int_x<377 then
                    y <= "0000000011100100"; -- 228
                elsif int_x<386 then
                    y <= "0000000011100110"; -- 230
                elsif int_x<396 then
                    y <= "0000000011101000"; -- 232
                elsif int_x<406 then
                    y <= "0000000011101001"; -- 233
                elsif int_x<416 then
                    y <= "0000000011101011"; -- 235
                elsif int_x<426 then
                    y <= "0000000011101100"; -- 236
                elsif int_x<436 then
                    y <= "0000000011101110"; -- 238
                elsif int_x<446 then
                    y <= "0000000011101111"; -- 239
                elsif int_x<456 then
                    y <= "0000000011110000"; -- 240
                elsif int_x<466 then
                    y <= "0000000011110001"; -- 241
                elsif int_x<476 then
                    y <= "0000000011110010"; -- 242
                elsif int_x<486 then
                    y <= "0000000011110011"; -- 243
                elsif int_x<496 then
                    y <= "0000000011110100"; -- 244
                elsif int_x<506 then
                    y <= "0000000011110101"; -- 245
                elsif int_x<515 then
                    y <= "0000000011110110"; -- 246
                elsif int_x<525 then
                    y <= "0000000011110111"; -- 247
                elsif int_x<535 then
                    y <= "0000000011110111"; -- 247
                elsif int_x<545 then
                    y <= "0000000011111000"; -- 248
                elsif int_x<555 then
                    y <= "0000000011111000"; -- 248
                elsif int_x<565 then
                    y <= "0000000011111001"; -- 249
                elsif int_x<575 then
                    y <= "0000000011111001"; -- 249
                elsif int_x<585 then
                    y <= "0000000011111010"; -- 250
                elsif int_x<595 then
                    y <= "0000000011111010"; -- 250
                elsif int_x<605 then
                    y <= "0000000011111011"; -- 251
                elsif int_x<615 then
                    y <= "0000000011111011"; -- 251
                elsif int_x<625 then
                    y <= "0000000011111011"; -- 251
                elsif int_x<635 then
                    y <= "0000000011111100"; -- 252
                elsif int_x<644 then
                    y <= "0000000011111100"; -- 252
                elsif int_x<654 then
                    y <= "0000000011111100"; -- 252
                elsif int_x<664 then
                    y <= "0000000011111100"; -- 252
                elsif int_x<674 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<684 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<694 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<704 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<714 then
                    y <= "0000000011111101"; -- 253
                elsif int_x<724 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<734 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<744 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<754 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<764 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<773 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<783 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<793 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<803 then
                    y <= "0000000011111110"; -- 254
                elsif int_x<813 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<823 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<833 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<843 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<853 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<863 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<873 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<883 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<893 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<902 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<912 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<922 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<932 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<942 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<952 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<962 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<972 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<982 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<992 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1002 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1012 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1022 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1031 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1041 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1051 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1061 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1071 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1081 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1091 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1101 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1111 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1121 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1131 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1141 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1151 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1160 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1170 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1180 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1190 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1200 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1210 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1220 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1230 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1240 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1250 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1260 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1270 then
                    y <= "0000000011111111"; -- 255
                elsif int_x<1280 then
                    y <= "0000000011111111"; -- 255
                else
                    y <= "0000000100000000"; -- 256
                end if;

            end process tanh_process;

        end architecture rtl;
        """
        # noinspection PyTypeChecker
        data = float_values_to_fixed_point(
            np.linspace(-5, 5, 259).tolist(), total_bits=16, frac_bits=8
        )
        tanh = Tanh(x=data)
        tanh_code = tanh()
        tanh_code_str = ""
        for line in tanh_code:
            tanh_code_str += line + "\n"
        self.check_generated_code(expected_code, tanh_code_str)
        # clean each file from empty lines and lines which are just comment
