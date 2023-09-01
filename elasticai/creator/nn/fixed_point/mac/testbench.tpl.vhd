----------------------------------------------------------------------------------
-- Company:
-- Engineer:
--
-- Create Date: 08/21/2023 09:45:40 AM
-- Design Name:
-- Module Name: test_fxp_mac - Behavioral
-- Project Name:
-- Target Devices:
-- Tool Versions:
-- Description:
--
-- Dependencies:
--
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
--
----------------------------------------------------------------------------------


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity testbench_fxp_mac is
    generic(
        VECTOR_WIDTH : integer := 2;
        TOTAL_WIDTH   : integer := 4;
        FRAC_WIDTH   : integer := 2
    );
--  Port ( );
end testbench_fxp_mac;

architecture Behavioral of testbench_fxp_mac is
    signal reset : std_logic;
    signal next_sample : std_logic;
    signal x1 : signed (TOTAL_WIDTH-1 downto 0);
    signal x2 : signed (TOTAL_WIDTH-1 downto 0);
    signal sum :  signed(TOTAL_WIDTH-1 downto 0);
    signal done :  std_logic;

    file input_buf : text;  -- text is keyword
    file output_buf : text;  -- text is keyword

begin
    UUT : entity work.fxp_MAC_RoundToEven generic map(VECTOR_WIDTH => VECTOR_WIDTH, TOTAL_WIDTH=>TOTAL_WIDTH, FRAC_WIDTH => FRAC_WIDTH)
    port map (reset => reset, next_sample => next_sample, x1 => x1, x2 => x2, sum => sum, done => done);




    testbench_1 : process
    variable read_col_from_input_buf : line; -- read lines one by one from input_buf
    variable write_col_to_output_buf : line; -- line is keyword
    variable reset_val, next_sample_val : std_logic; -- to save col1 and col2 values of 1 bit
    variable x1_val, x2_val : std_logic_vector(TOTAL_WIDTH-1 downto 0); -- to save col3 value of 2 bit
    variable val_SPACE : character;  -- for spaces between data in file

    begin
    file_open(input_buf, "${input_file}",  read_mode);
    file_open(output_buf, "${output_file}",  write_mode);


    write(write_col_to_output_buf, string'("START_SIM"));
    writeline(output_buf, write_col_to_output_buf);
    write(write_col_to_output_buf, string'("reset next_sample x1 x2 sum done"));
    writeline(output_buf, write_col_to_output_buf);

    readline(input_buf, read_col_from_input_buf); --read headerrow and throw away

    while not endfile(input_buf) loop

          -- reading input
          readline(input_buf, read_col_from_input_buf);
          read(read_col_from_input_buf, reset_val);
          read(read_col_from_input_buf, val_SPACE);           -- read in the space character
          read(read_col_from_input_buf, next_sample_val);
          read(read_col_from_input_buf, val_SPACE);           -- read in the space character
          read(read_col_from_input_buf, x1_val);
          read(read_col_from_input_buf, val_SPACE);           -- read in the space character
          read(read_col_from_input_buf, x2_val);

          --connect input to signals
          reset <= reset_val;
          next_sample <= next_sample_val;
          x1 <= signed(x1_val);
          x2 <= signed(x2_val);

          -- writing output
          write(write_col_to_output_buf, reset_val);
          write(write_col_to_output_buf, string'(" "));
          write(write_col_to_output_buf, next_sample_val);
          write(write_col_to_output_buf, string'(" "));
          write(write_col_to_output_buf, x1_val);
          write(write_col_to_output_buf, string'(" "));
          write(write_col_to_output_buf, x2_val);
          write(write_col_to_output_buf, string'(" "));
          write(write_col_to_output_buf, std_logic_vector(sum));
          write(write_col_to_output_buf, string'(" "));
          write(write_col_to_output_buf, done);
          writeline(output_buf, write_col_to_output_buf);

          wait for 20 ns;
    end loop;
    write(write_col_to_output_buf, string'("END_SIM"));
    writeline(output_buf, write_col_to_output_buf);
    file_close(output_buf);
    file_close(input_buf);
    wait;
    end process;

end Behavioral;
