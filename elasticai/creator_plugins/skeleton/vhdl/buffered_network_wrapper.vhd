library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity buffered_network_wrapper is
  generic (
    DATA_IN_WIDTH : positive;
    DATA_IN_DEPTH : positive;
    DATA_OUT_WIDTH : positive;
    DATA_OUT_DEPTH : positive;
    STRIDE : positive;
    KERNEL_SIZE: positive
  );

  port (
    signal clk : in std_logic;
    signal rst : in std_logic;
    signal d_in : in std_logic_vector(8 - 1 downto 0);
    signal d_out : out std_logic_vector(8 - 1 downto 0);
    signal done : out std_logic;
    signal enable : in std_logic;
    signal wr: in std_logic;
    signal address: in std_logic_vector(15 downto 0)
    );
end entity;


architecture rtl of buffered_network_wrapper is
  signal run_computation : std_logic := '0';
  signal has_been_enabled : std_logic := '0';
  signal internal_done : std_logic := '0';
  signal in_buffer_valid_out : std_logic;
  signal in_buffer_d_out : std_logic_vector(DATA_IN_WIDTH*KERNEL_SIZE - 1 downto 0);
  signal network_d_out : std_logic_vector(DATA_OUT_WIDTH - 1 downto 0);
  signal network_valid_out : std_logic;
  signal address_i : integer range 0 to 2**16-1;
  signal network_rd_address : std_logic_vector(address'range);

  signal network_wr_address : std_logic_vector(address'range);

  constant OUT_DEPTH_BYTES : integer := size_in_bytes(DATA_OUT_WIDTH)*DATA_OUT_DEPTH;
  constant IN_DEPTH_BYTES : integer := size_in_bytes(DATA_IN_WIDTH)*DATA_IN_DEPTH;

  signal network_wr_address_i : integer range 0 to IN_DEPTH_BYTES - 1;
  signal network_rd_address_i : integer range 0 to OUT_DEPTH_BYTES - 1;
  begin

  update_run_computation:
  process (clk, rst) is
  begin
    if rst = '1' then
      has_been_enabled <= '0';
    elsif rising_edge(clk) then
      if enable = '1' then
        has_been_enabled <= '1';
      end if;
    end if;
  end process; 

  run_computation <= (has_been_enabled or enable) and not internal_done;
  done <= internal_done;
  address_i <= to_integer(unsigned(address));

  
  network_wr_address_i <= min_fn(IN_DEPTH_BYTES - 1, address_i);
  network_wr_address <= std_logic_vector(to_unsigned(network_wr_address_i, network_wr_address'length));

  network_rd_address_i <= min_fn(OUT_DEPTH_BYTES - 1, address_i);
  network_rd_address <= std_logic_vector(to_unsigned(network_rd_address_i, network_rd_address'length));


  input_buffer : entity work.addressable_input_buffer(rtl)
  generic map (
   DATA_WIDTH => DATA_IN_WIDTH,
   DATA_DEPTH => DATA_IN_DEPTH,
   DATA_OUT_DEPTH => KERNEL_SIZE,
   STRIDE => STRIDE
  )
  port map (
    write_enable => wr,
    address => address,
    d_in => d_in,
    clk => clk,
    rst => rst,
    ready_in => run_computation,
    valid_out => in_buffer_valid_out,
    d_out => in_buffer_d_out
  );

  network : entity work.network(rtl)
  port map (
    d_in => in_buffer_d_out,
    valid_in => run_computation,
    clk => clk,
    rst => rst,
    d_out => network_d_out,
    valid_out => network_valid_out
  );

  output_buffer : entity work.addressable_output_buffer(rtl)
  generic map (
    DATA_WIDTH => DATA_OUT_WIDTH,
    DATA_DEPTH => DATA_OUT_DEPTH,
    DATA_IN_DEPTH => 1
  )
  port map (
    clk => clk,
    rst => rst,
    valid_in => network_valid_out,
    address => address,
    d_in => network_d_out,
    valid_out => internal_done,
    d_out => d_out
  );
  

end architecture;
