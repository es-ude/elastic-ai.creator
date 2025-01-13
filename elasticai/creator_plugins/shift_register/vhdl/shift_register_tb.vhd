library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;

entity shift_register_tb is
end entity;

architecture behav of shift_register_tb is

    signal clk : std_logic := '1';
    constant KERNEL_SIZE : natural := 3;
    constant NUM_CHANNELS : natural := 3;
    constant NUM_STEPS : natural := 1;
    signal rst : std_logic := '0';
    signal enable : std_logic := '0';
    signal one : std_logic := '1';
    signal zero : std_logic := '0';
    constant TOTAL_LENGTH : natural := (KERNEL_SIZE + NUM_STEPS - 1 ) * NUM_CHANNELS;
    signal x : std_logic_vector(NUM_CHANNELS - 1 downto 0) := b"000";
    signal y : std_logic_vector(KERNEL_SIZE*NUM_CHANNELS - 1 downto 0) := (others => '0');
    signal valid_out : std_logic;
    signal sample_id : std_logic_vector(3 - 1 downto 0) := (others => '0');
    type input_t is array(0 to 4) of std_logic_vector(2 downto 0);
    constant input: input_t := (b"000", b"111", b"011", b"101", b"010");
    constant MAX_SIM_STEPS : positive := 31;
    signal sim_steps : std_logic_vector(5 - 1 downto 0) := (others => '0');
    type expected_t is array(0 to 4) of std_logic_vector(8 downto 0);
    signal int_sample_id : integer range 0 to 2**sample_id'length - 1 := 0;
    signal int_sim_step : integer range 0 to 2**sim_steps'length - 1 := 0;
    signal current_expected : std_logic_vector(8 downto 0) := (others => 'U');
    signal sample_enable : std_logic := '0';
    signal expected_id : std_logic_vector(3 - 1 downto 0) := (others => '0');
    signal int_expected_id : integer range 0 to 2**3 - 1;
    signal valid_output_id : std_logic_vector(2 - 1 downto 0) := (others => '0');
    signal int_valid_output_id : integer range 0 to 2;
    signal valid_output_id_enable : std_logic := '0';
    signal start : std_logic := '0';
    constant expected : expected_t := ( b"000000000",
       b"000000000",
     b"000000111",
    b"000111011", b"111011101"
    );
    begin
        int_sample_id <= to_integer(unsigned(sample_id));
        int_sim_step <= to_integer(unsigned(sim_steps));
        int_expected_id <= to_integer(unsigned(expected_id));
        x <= input(int_sample_id);
        current_expected <= expected(int_expected_id);
        int_valid_output_id <= to_integer(unsigned(valid_output_id));
        valid_output_id_enable <= '1' when valid_out = '1' and int_valid_output_id < 2 else '0';
        enable <= '1' when int_sample_id < input'right and start = '1' else '0';

        dut_i : entity work.shift_register(rtl)
            generic map (
              DATA_WIDTH => NUM_CHANNELS,
              NUM_POINTS => KERNEL_SIZE
            )
            port map (
                d_in => x,
                d_out => y,
                clk => clk,
                valid_in => enable,
                valid_out => valid_out,
                rst => rst
            );

        sim_step_counter_i : entity work.counter(rtl)
            generic map(
              MAX_VALUE => MAX_SIM_STEPS
            )
            port map (
              d_out => sim_steps,
              rst => zero,
              enable => one,
              clk => clk
            );

        expected_id_counter_i :entity work.counter(rtl)
          generic map (
            MAX_VALUE => expected'right
          )
          port map (
            d_out => expected_id,
            rst => zero,
            enable => enable,
            clk => clk
          );

        valid_out_counter_i : entity work.counter(rtl)
          generic map (
            MAX_VALUE => 2
          )
          port map (
            d_out => valid_output_id,
            rst => zero,
            enable => valid_output_id_enable,
            clk => clk
          );

        sample_counter_i : entity work.counter(rtl)
            generic map (
              MAX_VALUE => input'right
            )
            port map (
                d_out => sample_id,
                rst => rst,
                enable => enable,
                clk => clk
            );

        check_output: process (clk) is
        begin
            -- if rising_edge(clk) then
            if enable = '1' then
                assert current_expected = y report "expected " & to_string(current_expected) & " but was " & to_string(y) severity error;
            end if;
        end process;

       feed_data: process is
       begin
           wait until int_sim_step = 2;
           rst <= '1';
           wait until int_sim_step = 4;
           rst <= '0';
           wait until int_sim_step = 6;
           rst <= '1';
           wait until int_sim_step = 8;
           rst <= '0';
           wait until int_sim_step = 9;
           start <= '1';
           for i in 0 to 4 loop
               if i = 4 then
                   assert valid_out = '1' report "expected valid out to be '1', but was " & to_string(valid_out);
               end if;
               wait until int_sim_step = i + 10;
           end loop;
            wait until int_sim_step = MAX_SIM_STEPS;
            finish;

       end process;

    clock: process is begin
        clk <= not clk;
        wait for 10 ps;
    end process;
        

end architecture;
