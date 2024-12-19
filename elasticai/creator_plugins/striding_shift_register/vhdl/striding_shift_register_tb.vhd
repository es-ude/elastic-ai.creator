library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

entity striding_shift_register_tb is
end entity;

architecture behav of striding_shift_register_tb is
    signal clk : std_logic := '1';
    signal rst : std_logic := '0';
    constant KERNEL_SIZE : natural := 3;
    constant NUM_CHANNELS : natural := 3;
    constant NUM_STEPS : natural := 1;
    constant TOTAL_LENGTH : natural := (KERNEL_SIZE + NUM_STEPS - 1) * NUM_CHANNELS;
    signal x : std_logic_vector(NUM_CHANNELS - 1 downto 0) := (others => '0');
    type input_data_t is array(0 to 5) of std_logic_vector(2 downto 0);
    signal input_data : input_data_t := (b"000", b"111", b"001", b"111", b"010", b"111");
    signal y : std_logic_vector(KERNEL_SIZE*NUM_CHANNELS - 1 downto 0) := (others => '0');
    signal counter : integer := 0;
    type expected_t is array(0 to 4) of std_logic_vector(9 - 1 downto 0);
    constant expected : expected_t := (
        b"UUUUUUUUU",b"UUUUUU000",b"UUUUUU000", b"UUU000001",
                                           b"UUU000001");
    signal expected_counter : integer := 0;
    signal register_valid : std_logic := '0';
    signal input_valid : std_logic := '1';
    begin

        dut_i : entity work.striding_shift_register(rtl)
            generic map (
              DATA_WIDTH => NUM_CHANNELS,
              NUM_POINTS => KERNEL_SIZE,
              STRIDE => 2
            )
            port map (
                d_in => x,
                d_out => y,
                clk => clk,
                valid_out => register_valid,
                valid_in => input_valid,
                rst => rst
            );

        clock: process is begin
            clk <= not clk;
            wait for 10 ps;
        end process;


        counting: process (clk) is
            variable  sim_steps : integer := 0;
        begin
            if rising_edge(clk) then
                if sim_steps = 20 then
                    finish;
                else
                    sim_steps := sim_steps + 1;
                    if counter < input_data'length - 1 then
                        input_valid <= '1';
                        counter <= counter + 1;
                    else
                        input_valid <= '0';
                    end if;
                    if expected_counter < 2*expected'length then
                        expected_counter <= expected_counter + 1;
                    end if;
                end if;
            end if;

        end process;

        x <= input_data(counter);

        reporting: process (clk) is begin
                    if rising_edge(clk) then
                        if expected_counter < expected'length  then
                            assert expected(expected_counter) = y report "expected " & to_string(expected(expected_counter)) & " but was " & to_string(y) severity error;
                            assert register_valid = '0' report "expected output invalid before register is full" severity error;
                        elsif expected_counter = (expected'length ) then
                            assert register_valid = '1' report "output valid for full register" severity error;
                            assert b"000001010" = y report "expected 000001010 but was " & to_string(y) severity error;
                        else
                            assert register_valid = '1' report "output expected to stay valid but was invalid" severity error;
                        end if;
                end if;

        end process;


end architecture;
