library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
use std.env.finish;


entity testbench_fxp_mac is

end testbench_fxp_mac;

architecture Behavioral of testbench_fxp_mac is
    constant VECTOR_WIDTH : integer := 2;
    constant TOTAL_WIDTH : integer := 4;
    constant FRAC_WIDTH : integer := 2;
    constant total_clock_cycles: integer := 4;
    signal clock_period : time := 2 ps;
    signal clock : std_logic;
    signal reset : std_logic;
    signal next_sample : std_logic;
    signal x1 : signed (TOTAL_WIDTH-1 downto 0) := (others => '0');
    signal x2 : signed (TOTAL_WIDTH-1 downto 0) := (others => '0');
    signal sum :  signed(TOTAL_WIDTH-1 downto 0);
    signal done :  std_logic;

    type input_array_t is array (0 to 1) of signed(TOTAL_WIDTH-1 downto 0);
    signal x1_values : input_array_t := ($x1);
    signal x2_values : input_array_t := ($x2);



begin
    UUT : entity work.fxp_MAC_RoundToEven generic map(VECTOR_WIDTH => VECTOR_WIDTH, TOTAL_WIDTH=>TOTAL_WIDTH, FRAC_WIDTH => FRAC_WIDTH)
    port map (reset => reset, next_sample => next_sample, x1 => x1, x2 => x2, sum => sum, done => done);

    clock_process: process
    begin
        clock <= '0';
        wait for clock_period/2;
        clock <= '1';
        wait for clock_period/2;
    end process;

    next_sample <= clock;

    testbench_1 : process(clock)
      variable iteration_id : integer := 1;
      variable reset_performed : std_logic := '0';
    begin
        if rising_edge(clock) and reset_performed = '0' then
            reset <= '0';
            reset_performed := '1';
        end if;
        if falling_edge(clock) then
            if reset_performed = '0' then
                reset <= '1';
            elsif iteration_id < 2 then
                x1 <= x1_values(iteration_id);
                x2 <= x2_values(iteration_id);
                iteration_id := iteration_id + 1;
            elsif done = '1' then
                report to_bstring(sum);
                finish;
            else
                iteration_id := iteration_id + 1;
            end if;
        end if;
    end process;

end Behavioral;
