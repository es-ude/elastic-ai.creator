library ieee;
    use ieee.std_logic_1164.all;
    use ieee.numeric_std.all;
entity NaiveLUTConv is
    generic (
        INPUT_WIDTH : integer := 1;
        OUTPUT_WIDTH : integer := 1
    );
    port (
        x : in std_logic_vector(INPUT_WIDTH-1 downto 0);
        y : out std_logic_vector(OUTPUT_WIDTH-1 downto 0)
    );
end entity NaiveLUTConv;
architecture rtl of NaiveLUTConv is
begin
    y <="1" when x="00" else
"0" when x="01" else
"1" when x="10" else
"0" when x="11" ;
end architecture rtl;
