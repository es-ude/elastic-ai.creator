library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use work.bus_package.all;
    
entity activations_bram is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(5-1 downto 0);
        data : out bus_array_4_8
    );
end entity activations_bram;

architecture rtl of activations_bram is
    type activations_bram_array_t is array (0 to 31) of std_logic_vector(8-1 downto 0);
    signal rom : activations_bram_array_t := ("00000001","00000001","00000001","00000000","00000000","11111110","00000000","00010011","00100000","00001111","11110111","11101100","11101100","11101111","11110001","11110011","11110101","11110111","11111000","11111011","11111101","11111111","00000000","00000000","00000000","00000000","00000000","00000000","11111111","00000000","00000000","00000000");
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "auto";
begin
    ROM_process: process(clk)
    begin
        if rising_edge(clk) then
            if (en = '1') then
                data <= bus_array_4_8((ROM(conv_integer(addr) to conv_integer(addr)+3)));
            end if;
        end if;
    end process ROM_process;
end architecture rtl;