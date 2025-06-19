library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;

entity dynamic_padder is

  generic (
    DATA_WIDTH : positive := 1;
    DATA_DEPTH : positive 
  );

  port (
    d_in : in std_logic_vector(DATA_WIDTH*DATA_DEPTH - 1 downto 0);
    d_out : out std_logic_vector(8 - 1 downto 0);
    clk : in std_logic;
    ready_in: in std_logic;
  );

end entity;

architecture rtl of dynamic_padder is


begin

  process (clk) is

  begin


  end process;


end architecture;

