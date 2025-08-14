library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;


entity dynamic_padding_remover is

  generic (
    DATA_WIDTH : positive
  );

  port (
    d_in : in std_logic_vector(8 - 1 downto 0);
    d_out : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    clk : in std_logic;
    ready_in : in std_logic;
    ready_out : out std_logic;
    valid_in : in std_logic;
    valid_out : out std_logic;
    rst : in std_logic
  );
end entity;

architecture rtl of dynamic_padding_remover is
  constant NUM_BYTES : positive := size_in_bytes(DATA_WIDTH);
  constant REMAINDER : natural := NUM_BYTES*8 - DATA_WIDTH;
  signal bytes : std_logic_vector(NUM_BYTES * 8 - 1 downto 0);
begin
  ready_out <= ready_in;

  shift_reg: entity work.striding_shift_register(rtl)
    generic map (
      DATA_WIDTH => 8,
      NUM_POINTS => NUM_BYTES,
      STRIDE => NUM_BYTES
    )
    port map (
      d_in => d_in,
      d_out => bytes,
      valid_in => valid_in,
      clk => clk,
      rst => rst,
      valid_out => valid_out
    );

  d_out <= bytes(NUM_BYTES*8-1 - REMAINDER downto 0);

end architecture;

