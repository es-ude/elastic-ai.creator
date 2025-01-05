library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity ${name}_tb is
end ${name}_tb;

architecture testbench of ${name}_tb is
    signal clock : std_logic := '0';
    signal enable : std_logic := '0';
    signal x : std_logic_vector(${data_width - 1} downto 0);
    signal y : std_logic_vector(${data_width - 1} downto 0);
    signal done : std_logic;

    constant CLK_PERIOD : time := 10 ns;

    type t_inputs is array (integer range <>) of std_logic_vector(${data_width - 1} downto 0);
    constant test_inputs : t_inputs := (
        -- Add test inputs here
        (others => '0')
    );

begin
    clock_process : process
    begin
        clock <= not clock;
        wait for CLK_PERIOD / 2;
    end process;

    -- Add stimulus process for testing depthwise convolution logic

    uut: entity ${work_library_name}.${name}
        port map (
            enable => enable,
            clock  => clock,
            x      => x,
            y      => y,
            done   => done
        );

end testbench;
