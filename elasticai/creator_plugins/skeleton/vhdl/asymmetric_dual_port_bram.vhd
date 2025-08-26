library ieee;
use ieee.std_logic_1164.all;
use ieee.NUMERIC_STD.all;
use work.skeleton_pkg.all;

-- The implementation is basically copied from the
-- xilinx synthesis user guide UG901 (v2021.2)
-- It features a pipelined read for higher clock rates
-- Data is stored in reversed word order

entity asymmetric_dual_port_bram is
  generic (
    WRITE_DATA_WIDTH: positive := 4;
    WRITE_ADDRESS_WIDTH: positive := 10;
    WRITE_SIZE: positive := 1024;
    READ_DATA_WIDTH: positive := 16;
    READ_ADDRESS_WIDTH: positive := 8;
    READ_SIZE: positive := 256
  );
  port (
    signal read_clk : in std_logic;
    signal read_address : in std_logic_vector(READ_ADDRESS_WIDTH - 1 downto 0);
    signal read_enable : in std_logic;
    signal d_out : out std_logic_vector(READ_DATA_WIDTH - 1 downto 0);

    signal write_clk : in std_logic;
    signal write_address : in std_logic_vector(WRITE_ADDRESS_WIDTH - 1 downto 0);
    signal write_enable : in std_logic;
    signal d_in : in std_logic_vector(WRITE_DATA_WIDTH - 1 downto 0)
  );
end entity;

architecture rtl of asymmetric_dual_port_bram is

  constant MIN_DATA_WIDTH : integer := fmin(READ_DATA_WIDTH, WRITE_DATA_WIDTH);
  constant MAX_DATA_WIDTH : integer := fmax(READ_DATA_WIDTH, WRITE_DATA_WIDTH);
  constant MAX_SIZE : integer := fmax(READ_SIZE, WRITE_SIZE);
  constant RATIO : integer := MAX_DATA_WIDTH / MIN_DATA_WIDTH;


  function compute_int_address(i: integer; address: std_logic_vector) return integer is
    variable tmp: std_logic_vector(log2(RATIO) - 1 downto 0) := (others => '0');
    variable tmp2: std_logic_vector(address'length + tmp'length - 1 downto 0) := (others => '0');
  begin
    tmp := std_logic_vector(to_unsigned(i, tmp'length));
    tmp2 := address & tmp;
    return to_integer(unsigned(tmp2));
  end function;

  type ram_t is array (natural range 0 to MAX_SIZE - 1) of std_logic_vector(MIN_DATA_WIDTH - 1 downto 0);

  signal my_ram : ram_t := (others => (others => 'X')); 
  signal read_data : std_logic_vector(READ_DATA_WIDTH - 1 downto 0) := (others => '0');
  signal read_register : std_logic_vector(READ_DATA_WIDTH - 1 downto 0) := (others => '0');
  signal read_address_i : integer := 0;
  signal write_address_i : integer := 0;
begin

  read_address_i <= to_integer(unsigned(read_address));
  write_address_i <= to_integer(unsigned(write_address));
  

  read_wider: if MAX_DATA_WIDTH = READ_DATA_WIDTH generate

    process (write_clk) is
    begin
      if rising_edge(write_clk) then
        if write_enable = '1' then
          my_ram(write_address_i) <= d_in;
        end if;
      end if;
    end process;

    process (read_clk) is
    begin
      if rising_edge(read_clk) then
        for i in 0 to RATIO - 1 loop
          if read_enable = '1' then
            read_data((i + 1) * MIN_DATA_WIDTH - 1 downto i * MIN_DATA_WIDTH) <= my_ram(compute_int_address(i, read_address));
          end if;
        end loop;
        read_register <= read_data;
      end if;
    end process;

    d_out <= read_register;

  end generate;


  write_wider: if MAX_DATA_WIDTH = WRITE_DATA_WIDTH generate

    process (read_clk) is
    begin
      if rising_edge(read_clk) then
        if read_enable = '1' then
          read_data <= my_ram(read_address_i);
        end if;
        read_register <= read_data;
      end if;
    end process;

    process (write_clk) is
    begin
      if rising_edge(write_clk) then
        for i in 0 to RATIO - 1 loop
          if write_enable = '1' then
            my_ram(compute_int_address(i, write_address)) <= d_in((i+1)*MIN_DATA_WIDTH - 1 downto i * MIN_DATA_WIDTH);
          end if;
        end loop;
      end if;
    end process;

    d_out <= read_register;
    
  end generate;

end architecture;
