library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

entity sliding_window_tb is
end entity;

architecture behav of sliding_window_tb is
    signal clk : std_logic := '0';
    constant KERNEL_SIZE : natural := 3;
    constant NUM_CHANNELS : natural := 3;
    constant NUM_STEPS : natural := 1;
    constant rst : std_logic := '0';
    constant TOTAL_LENGTH : natural := (KERNEL_SIZE + NUM_STEPS - 1 ) * NUM_CHANNELS;
    signal x : std_logic_vector(12 - 1 downto 0) := b"111011101010";
    signal y : std_logic_vector(3 - 1 downto 0) := (others => '0');
    signal valid_in : std_logic := '1';
    begin

      
        dut_i : entity work.sliding_window(rtl)
            generic map (
                INPUT_WIDTH => 12,
                OUTPUT_WIDTH => 3
            )
            port map (
                d_in => x,
                d_out => y,
                clk => clk,
                valid_in => valid_in,
                rst => rst
            );

        process is begin
            wait for 10 ns;
            clk <= not clk;
        end process;

        feed_data: process (clk) is
            type expected_t is array (0 to 4 - 1) of std_logic_vector(3 - 1 downto 0);
            constant expected : expected_t :=
                (b"111", b"110", b"101", b"011");
            variable counter : natural := 0;
        begin

            if rising_edge(clk) then
                assert expected(counter) = y report "expected " & to_string(expected(counter)) &
                    " but was " & to_string(y) severity error;
                if counter < expected'length - 1 then
                    counter := counter + 1;
                else
                    finish;
                end if;
            end if;
        end process;
end architecture;
