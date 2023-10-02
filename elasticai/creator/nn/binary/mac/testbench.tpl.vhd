library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;
use std.env.finish;


entity $name is

end;

architecture Behavioral of $name is
    constant TOTAL_WIDTH : integer := $total_width;
    constant total_clock_cycles: integer := 4;
    signal clock_period : time := 2 ps;
    signal clock : std_logic := '0';
    signal reset : std_logic := '0';
    subtype input_t is std_logic_vector(TOTAL_WIDTH-1 downto 0);
    signal next_sample : std_logic;
    signal x1 : input_t := (others => '0');
    signal x2 : input_t := (others => '0');
    signal sum :  std_logic;
    signal done :  std_logic;

    signal x1_values : input_t := $x1;
    signal x2_values : input_t := $x2;



begin
    UUT : entity work.${uut_name}
    port map (reset => reset, next_sample => next_sample, x1 => x1, x2 => x2, sum => sum, done => done);

    next_sample <= clock;
    clock <= not clock after clock_period/2;

    testbench_1 : process(clock, done)
      variable iteration_id : integer := 0;
      variable reset_performed : std_logic := '0';
      variable value_id : integer := 0;
      constant max_iterations : integer := 5;
    begin
        if rising_edge(clock) then
            if iteration_id = 0 then
                x1 <= x1_values;
                x2 <= x2_values;
            else
                report to_string(sum);
                finish;
            end if;
            iteration_id := iteration_id + 1;
        end if;
    end process;

end Behavioral;
