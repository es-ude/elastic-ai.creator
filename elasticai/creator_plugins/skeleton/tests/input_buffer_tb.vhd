library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.env.all;
use ieee.math_real.all;
library vunit_lib;
context vunit_lib.vunit_context;

entity input_buffer_tb is
  generic (
    runner_cfg: string
  );
end entity;

architecture behav of input_buffer_tb is


  constant DATA_DEPTH : integer := 6;
  constant DATA_WIDTH : integer := 12;
  constant DATA_OUT_DEPTH : integer := 1;
  constant STRIDE : integer := 1;
  signal write_enable : std_logic;

  subtype buffer_d_t is std_logic_vector(DATA_WIDTH - 1 downto 0);
  subtype byte_t is std_logic_vector(7 downto 0);
  type byte_array_t is array (natural range <>) of byte_t;
  signal buffer_d_out : buffer_d_t := (others => 'X');
  
  signal bytes_d_in : byte_t;


  signal read_enable : std_logic := '0';
  signal clk : std_logic := '0';
  signal rst : std_logic := '0';
  constant half_period : time := 1 fs;
  signal address : std_logic_vector(15 downto 0) := (others => '0');
  signal address_int : integer range 0 to 2**16 -1 := 0;
  signal valid_out : std_logic;
begin

    clocking : process is
    begin
        clk <= '0';
        wait for half_period;
        clk <= '1';
        wait for half_period;
    end process;
    
    address <= std_logic_vector(to_unsigned(address_int, address'length));

    input_buffer : entity work.addressable_input_buffer(rtl)
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            DATA_DEPTH => DATA_DEPTH,
            DATA_OUT_DEPTH => DATA_OUT_DEPTH,
            STRIDE => STRIDE
        )
        port map (
            write_enable => write_enable,
            address => address,
            d_in => bytes_d_in,
            d_out => buffer_d_out,
            ready_in => read_enable,
            valid_out => valid_out,
            clk => clk,
            rst => rst
        );
    
    stimulus : process is
        variable counter : integer := 0;
        procedure write_data_to_buffer(data: in byte_array_t) is
        begin
            for i in data'range loop
                bytes_d_in <= data(i);
                debug("writing to " & integer'image(i));
                address_int <= i;
                write_enable <= '1';
                wait until rising_edge(clk);
            end loop;
            write_enable <= '0';
        end procedure;
        
        procedure check_output(expected: in buffer_d_t) is
        begin
            read_enable <= '1';
            wait until rising_edge(clk);
            check_equal(buffer_d_out, expected);
            read_enable <= '0';
        end procedure;
    begin
        test_runner_setup(runner, runner_cfg);
        rst <= '1';
        wait until rising_edge(clk);
        
        rst <= '0';
        wait until rising_edge(clk);

        -- Test writing and reading from the buffer
        if run("can write 01_00 and read 00_01") then
            write_data_to_buffer((x"01", x"00"));
            wait until rising_edge(clk); -- wait until next rising edge for ram value to become available at d_out
            check_output(b"0000_0000_0001");
        end if;

        if run("can write DE_AD_BE_EF read D_DE_F_BE") then
            write_data_to_buffer((x"DE",  x"AD", x"BE", x"EF"));
            check_output(x"D_DE"); -- don't need to wait here cause we've been writing twice
            check_output(x"F_BE");
        end if;

        if run("fill buffer, read full buffer and check control signals") then
            write_data_to_buffer((
                x"01",
                x"00",
                x"02",
                x"00",
                x"03",
                x"00",
                x"04",
                x"00",
                x"05",
                x"00",
                x"06",
                x"00"
            ));
            read_enable <= '1';
            while valid_out = '1' and counter < 10 loop
                counter := counter + 1;
                wait until rising_edge(clk);
            end loop;
            check_equal(counter, 6);
        end if;


    
        test_runner_cleanup(runner);
    end process;
end architecture;
