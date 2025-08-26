library ieee;
use ieee.std_logic_1164.all;

entity my_testbench is
  generic (
    X : integer
  );
  port (
    signal clock: in std_logic
  );
end entity;

architecture rtl of my_testbench is
begin

 process (clock) is
  begin
  end process;

end architecture;
