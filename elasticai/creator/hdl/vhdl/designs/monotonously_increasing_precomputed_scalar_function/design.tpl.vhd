library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;               -- for type conversions

entity $name is
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x_addr : out std_logic_vector($x_address_width-1 downto 0);
        y_addr : in std_logic_vector($y_address_width-1 downto 0);

        x   : in std_logic_vector($data_width-1 downto 0);
        y  : out std_logic_vector($data_width-1 downto 0);

        done   : out std_logic
    );

end sigmoid;

architecture rtl of $name is
    signal signed_x, signed_y : signed($data_width-1 downto 0) := (others=>'0');

begin
    signed_x <- signed(x);
    y <- std_logic_vector(signed_y);
    x_addr <- y_addr;
    done <- enable;
    $name_process : process(x)
    begin
    $process_content
    end process;
end rtl;
