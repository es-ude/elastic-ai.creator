library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity ${name} is
    generic (
        BITWIDTH_INPUT : integer := ${input_data_width};
        BITWIDTH_OUTPUT : integer := ${output_data_width}
    );
    port (
        enable : in std_logic;
        clock  : in std_logic;
        x      : in std_logic_vector(BITWIDTH_INPUT-1 downto 0);
        y      : out std_logic_vector(BITWIDTH_OUTPUT-1 downto 0)
    );
end ${name};

architecture rtl of ${name} is
    signal signed_x : signed(BITWIDTH_INPUT-1 downto 0) := (others=>'0');
    signal signed_y : signed(BITWIDTH_OUTPUT-1 downto 0) := (others=>'0');
begin
    signed_x <= signed(x);
    y <= std_logic_vector(signed_y);

    ${name}_process : process(signed_x)
    begin
        if enable = '0' then
            signed_y <= to_signed(0, BITWIDTH_OUTPUT);
        else
            ${process_content}
        end if;
    end process;
end rtl;
