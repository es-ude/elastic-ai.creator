library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
entity $name is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector($x_data_width-1 downto 0);
        y      : out std_logic_vector($y_data_width-1 downto 0)
    );
end $name;
architecture rtl of $name is
    signal signed_x : signed($x_data_width-1 downto 0) := (others=>'0');
    signal signed_y : signed($y_data_width-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);
    ${name}_process : process(x,clock,enable)
    begin
        if (enable = '0') then
            signed_y <= (others=>'0');
        elsif (rising_edge(clock)) then
            $process_content
        end if;
    end process;
end architecture;
