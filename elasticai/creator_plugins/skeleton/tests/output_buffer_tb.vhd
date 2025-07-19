library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
library vunit_lib;
context vunit_lib.vunit_context;
use work.skeleton_pkg.all;


entity output_buffer_tb is
  generic (
    runner_cfg: string
  );
end entity;

architecture behaviour of output_buffer_tb is
  type config_t is record
    DATA_WIDTH : integer;
    DATA_IN_DEPTH : integer;
    DATA_DEPTH : integer;
  end record;

  
  constant NUM_WRITES_A : integer := 4;

  function get_num_reads(config: config_t; num_writes: integer) return integer is
  begin
    return num_writes * size_in_bytes(config.DATA_WIDTH) * config.DATA_IN_DEPTH;
  end function;

  function left_in(config : config_t) return natural is
  begin
    return config.DATA_WIDTH*config.DATA_IN_DEPTH - 1;
  end function;

  function right_in(config: config_t) return natural is
  begin
    return 0;
  end function;


  constant config_a : config_t := (
    DATA_WIDTH => 2,
    DATA_IN_DEPTH => 1,
    DATA_DEPTH => 4
  );
  signal clk : std_logic := '0';
  signal rst : std_logic := '0';
  signal address : std_logic_vector(15 downto 0);
  signal address_int : integer range 0 to 2**16-1;


  signal d_in_a : std_logic_vector(left_in(config_a) downto right_in(config_a));
  signal d_out_a : std_logic_vector(7 downto 0);
  signal valid_out_a : std_logic;
  signal valid_in : std_logic;
  signal enable_timing_test : std_logic := '0';
  type result_t is array(0 to get_num_reads(config_a, NUM_WRITES_A) - 1) of std_logic_vector(7 downto 0);
  signal result : result_t := (others => (others => '0'));
  signal previous_address_int : integer range 0 to 2**16-1 := 0;
  signal fetching_result : std_logic := '0';
begin


  clk <= not clk after 1 fs;
  address <= std_logic_vector(to_unsigned(address_int, 16));

  dut_a: entity work.addressable_output_buffer(rtl)
  generic map (
    DATA_WIDTH => config_a.DATA_WIDTH,
    DATA_IN_DEPTH => config_a.DATA_IN_DEPTH,
    DATA_DEPTH => config_a.DATA_DEPTH  
  )
  port map (
    clk => clk,
    rst => rst,
    valid_in => valid_in,
    valid_out => valid_out_a,
    address => address,
     d_in => d_in_a,
    d_out => d_out_a
  );

  writer:
  process (clk, rst) is
    variable counter : integer := 0;
  begin
    if rst = '1' then
      counter := 0;
      valid_in <= '0';
    elsif rising_edge(clk) then
      valid_in <= '0';
      if enable_timing_test = '1' then
        if valid_out_a = '0' then
          valid_in <= '1';
          d_in_a <= std_logic_vector(to_unsigned(counter, d_in_a'length));
          counter := counter + 1;
        end if;
      else
        counter := 0;
      end if;
    end if;
  end process;

  update_address:
  process (clk, rst) is
  begin
    if rst = '1' then
      address_int <= 0;
    elsif rising_edge(clk) then
      if valid_out_a = '1' then
        if address_int < result_t'high then
          address_int <= address_int + 1;
        end if;
      end if;
    end if;
  end process;

  --- buffer is filled with number 0 1 2 3 
  -- rising edges         : t0    t1     t2     t3      t4      t5
  -- address for buffer   :  0     1      2      3       3       3
  -- address for result   :  0     0      1      2       3       3
  -- value of buffer d_out:  0     0      1      2       3       3
  -- value of result      :  0UUU  0UUU   0UUU   01UU    012U    0123
  -- fetching_result      :  H     H      H      H       H       L
  --                                                     ^
  --                                                     |
  --   fulfilling condition for setting fetching_result to L
  update_prev_addr:
  process (clk) is
    variable prev_prev_address : integer := 0;
  begin
    if rising_edge(clk) then
      if previous_address_int < 3 then
        fetching_result <= '1';
      else
        fetching_result <= '0';
      end if;
      previous_address_int <= address_int;
      prev_prev_address := previous_address_int;
    end if;
  end process;

  update_result:
  process (clk) is
  begin
    if rising_edge(clk) then
      if valid_out_a = '1' then
        result(previous_address_int) <= d_out_a;
      end if;
    end if;
  end process;


  stimuli:
  process is
    procedure wait_edge is
    begin
      wait until rising_edge(clk);
    end procedure;

  begin
    test_runner_setup(runner, runner_cfg);
    rst <= '1';
    wait_edge;
    rst <= '0';
    wait_edge;    


    if run("timing test writing values until finished writes last byte") then
      enable_timing_test <= '1';
      wait until fetching_result = '0';
      enable_timing_test <= '0';
      for i in result_t'range loop
        check_equal(to_integer(unsigned(result(i))), i);
      end loop; 
      wait_edge;
    end if;



    test_runner_cleanup(runner);
  end process;

end architecture;
