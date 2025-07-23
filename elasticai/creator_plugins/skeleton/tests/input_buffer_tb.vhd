library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
use work.skeleton_pkg.all;

package input_buffer_tb_pkg is

  type config_t is record
    data_width: positive;
    data_depth: positive;
    data_out_depth: positive;
    num_writes: natural;
    num_reads: natural;
    stride : positive;
  end record;

  type result_t is array(integer range <>) of std_logic_vector;
  function create_config(
    data_width: positive;
    data_depth: positive;
    data_out_depth: positive;
    stride: positive
  ) return config_t;
end package;

package body input_buffer_tb_pkg is


  function create_config(
        data_width: positive;
        data_depth: positive;
        data_out_depth: positive;
        stride : positive
    ) return config_t is
    variable result : config_t := (
        DATA_WIDTH => data_width,
        DATA_DEPTH => data_depth,
        DATA_OUT_DEPTH => data_out_depth,
        NUM_WRITES => 1,
        NUM_READS => 1,
        STRIDE => stride
    );
  begin
    result.NUM_WRITES := size_in_bytes(DATA_WIDTH)*DATA_DEPTH;
    result.NUM_READS := DATA_DEPTH/DATA_OUT_DEPTH;
    return result;
  end function;


end package body;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
use work.skeleton_pkg.all;
use work.input_buffer_tb_pkg.all;

entity buffer_client is
    generic (
        config: config_t
    );
    port (
        signal enable: in std_logic;
        signal done: out std_logic;
        signal rst: in std_logic;
        signal clk: in std_logic;
        signal data_input: out std_logic_vector(7 downto 0);
        signal address: out std_logic_vector(15 downto 0);
        signal buffer_valid: in std_logic;
        signal read_buffer: out std_logic;
        signal write_buffer: out std_logic;
        signal buffer_data : in std_logic_vector(config.data_width*config.data_out_depth - 1 downto 0);
        signal result: out result_t(0 to config.num_reads - 1)(config.data_width*config.data_out_depth - 1 downto 0)
    );
end entity;

architecture behaviour of buffer_client is

  signal address_int : integer range 0 to 2**16 -1 := 0;
  type tb_state_t is (starting, writing_data, writing_finished, fetching_result, finished);
  signal last_tb_state : tb_state_t := starting;
  signal tb_state : tb_state_t := starting;
    signal fetch_counter : integer range 0 to config.num_reads := 0;
begin
    address <= std_logic_vector(to_unsigned(address_int, address'length));
    done <= '1' when last_tb_state = finished else '0';
    read_buffer <= '1' when tb_state = fetching_result else '0';
    write_buffer <= '1' when last_tb_state = writing_data else '0';

    write_d_in:
    process (clk) is
        constant last_byte : integer := size_in_bytes(config.data_width) - 1;
        variable current_value_i : integer range 0 to 2**config.data_width - 1 := 0;
        variable byte_id : integer range 0 to size_in_bytes(config.data_width) - 1 := 0;
        variable current_value : std_logic_vector(size_in_bytes(config.data_width)*8 - 1 downto 0);
    begin
        if rising_edge(clk) then
            if rst = '1' then
                byte_id := 0;
                current_value_i := 0;
            elsif tb_state = writing_data then
                current_value := std_logic_vector(to_unsigned(current_value_i, 16));
                data_input <= current_value((byte_id + 1)*8 - 1 downto (byte_id)*8);
                
                if byte_id = last_byte then
                    current_value_i := current_value_i + 1;
                    byte_id := 0;
                else
                    byte_id := byte_id + 1;
                end if;
            end if;
        end if;
    end process;

    update_address:
    process (clk) is
    begin
        if rising_edge(clk) then
            address_int <= 0;
            if last_tb_state = writing_data then
                if address_int < config.NUM_WRITES - 1 then
                    address_int <= address_int + 1;
                end if;
            end if;
        end if;
    end process;

    update_state:
    process(clk) is
        variable write_count: integer range 0 to config.num_writes - 1 := 0;
    begin
        if rising_edge(clk) then
            if tb_state = starting then
                if enable = '1' then
                    tb_state <= writing_data;
                end if;
            elsif tb_state = writing_data then
                if write_count < config.num_writes - 1 then
                    write_count := write_count + 1;
                else
                    tb_state <= fetching_result;
                end if;
            elsif tb_state = fetching_result then
                if buffer_valid = '0' then
                    tb_state <= finished;
                end if;
            else
                tb_state <= finished;
            end if;
        end if;
    end process;

    update_last_state:
    process (clk) is
    begin
        if rising_edge(clk) then
            last_tb_state <= tb_state;
        end if;
    end process;

    fetch_result:
    process (clk) is
    begin
        if rising_edge(clk) then
            if tb_state = fetching_result then
                result(fetch_counter) <= buffer_data;
                fetch_counter <= fetch_counter + 1;
            end if;
        end if;
    end process;
    
end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
library vunit_lib;
context vunit_lib.vunit_context;
use work.skeleton_pkg.all;
use work.input_buffer_tb_pkg.all;


entity input_buffer_tb is
  generic (
    runner_cfg: string
  );
end entity;

architecture behav of input_buffer_tb is

  constant CFG : config_t := create_config(
    data_width => 12,
    data_depth => 6,
    data_out_depth => 1,
    stride => 1
  );
  signal enable_client : std_logic;
  signal write_enable : std_logic;
  signal read_enable : std_logic;
  subtype d_out_t is std_logic_vector(CFG.data_width*CFG.data_out_depth - 1 downto 0);

  signal d_in : std_logic_vector(7 downto 0);
  signal d_out : d_out_t;


  signal clk : std_logic := '0';
  signal rst : std_logic := '0';
  signal address : std_logic_vector(15 downto 0) := (others => '0');
  signal valid_out : std_logic;

  signal done : std_logic := '0';
  signal result : result_t(0 to CFG.num_reads - 1)(CFG.data_width * CFG.data_out_depth - 1 downto 0);

begin
    clk <= not clk after 1 fs;

    

    input_buffer : entity work.addressable_input_buffer(rtl)
        generic map (
            DATA_WIDTH => cfg.data_width,
            DATA_DEPTH => cfg.data_depth,
            DATA_OUT_DEPTH => cfg.data_out_depth,
            STRIDE => cfg.stride
        )
        port map (
            write_enable => write_enable,
            address => address,
            d_in => d_in,
            d_out => d_out,
            ready_in => read_enable,
            valid_out => valid_out,
            clk => clk,
            rst => rst
        );

    client : entity work.buffer_client(behaviour)
        generic map (
            config => CFG
        )
        port map (
            data_input => d_in,
            rst => rst,
            clk => clk,
            done => done,
            enable => enable_client,
            buffer_valid => valid_out,
            read_buffer => read_enable,
            write_buffer => write_enable,
            buffer_data => d_out,
            result => result,
            address => address
        );

    stimulus : process is
        variable counter : integer := 0;
    begin
        test_runner_setup(runner, runner_cfg);
        rst <= '1';
        wait until rising_edge(clk);
        
        rst <= '0';
        wait until rising_edge(clk);
        enable_client <= '1';
        wait until rising_edge(clk);

        while counter < 3000 loop
            if done = '1' then
                counter := 3000;
            end if;
            counter := counter + 1;
            wait until rising_edge(clk);
        end loop;
        
        for i in 0  to CFG.num_reads - 1 loop
            check_equal(to_integer(unsigned(result(i))), i);
        end loop;
    
        test_runner_cleanup(runner);
    end process;
end architecture;
