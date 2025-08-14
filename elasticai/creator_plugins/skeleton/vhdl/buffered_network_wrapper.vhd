library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity buffered_network_wrapper is
  generic (
    DATA_IN_WIDTH : positive := 1;
    DATA_IN_DEPTH : positive := 1;
    DATA_OUT_WIDTH : positive := 1;
    DATA_OUT_DEPTH : positive := 1;
    STRIDE : positive := 1;
    KERNEL_SIZE: positive := 1
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
  signal in_buffer_valid_out : std_logic;
  signal in_buffer_d_out : std_logic_vector(DATA_IN_WIDTH*KERNEL_SIZE - 1 downto 0);
  signal network_d_out : std_logic_vector(DATA_OUT_WIDTH - 1 downto 0);
  signal network_valid_out : std_logic;
  signal out_buffer_valid_out : std_logic;
  signal address_i : integer range 0 to 2**16-1;
  signal network_rd_address : std_logic_vector(address'range);

  signal network_wr_address : std_logic_vector(address'range);

  constant OUT_DEPTH_BYTES : integer := size_in_bytes(DATA_OUT_WIDTH)*DATA_OUT_DEPTH;
  constant IN_DEPTH_BYTES : integer := size_in_bytes(DATA_IN_WIDTH)*DATA_IN_DEPTH;
  signal network_wr_address_i : integer range 0 to IN_DEPTH_BYTES - 1;
  signal network_rd_address_i : integer range 0 to OUT_DEPTH_BYTES - 1;
  type network_state_t is (idle, start, running, finished);
  signal network_state : network_state_t := idle;
  signal rst_cycles : integer range 0 to 1 := 0;
  signal internal_rst : std_logic := '1';
  begin

  update_network_state:
  process(clk) is
  begin
    if rising_edge(clk) then
      if rst = '1' then
        network_state <= idle;
      elsif network_state = idle and enable = '1' then
        network_state <= start;
      elsif network_state = start then
        if rst_cycles = 1 then
          network_state <= running;
          rst_cycles <= 0;
        else
          rst_cycles <= rst_cycles + 1;
        end if;
      elsif network_state = running and out_buffer_valid_out = '1' then
        network_state <= finished;
      elsif network_state = finished and enable = '0' then
        network_state <= idle;
      end if;
    end if;
  end process;

    

  internal_rst <= '1' when network_state = start or rst = '1' else '0';
  run_computation <= '1' when network_state = running else '0';
  done <= '1' when network_state = finished else '0';
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
    rst => internal_rst,
    ready_in => run_computation,
    valid_out => in_buffer_valid_out,
    d_out => in_buffer_d_out
  );

  network : entity work.network(rtl)
  port map (
    d_in => in_buffer_d_out,
    valid_in => run_computation,
    clk => clk,
    rst => internal_rst,
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
    rst => internal_rst,
    valid_in => network_valid_out,
    address => address,
    d_in => network_d_out,
    valid_out => out_buffer_valid_out,
    d_out => d_out
  );
  

end architecture;
