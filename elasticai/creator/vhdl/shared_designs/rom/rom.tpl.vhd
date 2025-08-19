library ieee;
use ieee.numeric_std.all;
use ieee.std_logic_1164.all;

entity ${name} is
    generic (
        ROM_ADDR_WIDTH : integer := ${rom_addr_bitwidth};
        ROM_DATA_WIDTH : integer := ${rom_data_bitwidth}
    );
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(ROM_ADDR_WIDTH-1 downto 0);
        data : out std_logic_vector(ROM_DATA_WIDTH-1 downto 0)
    );
end entity ${name};
architecture rtl of ${name} is
    type ${name}_array_t is array (0 to 2**ROM_ADDR_WIDTH-1) of std_logic_vector(ROM_DATA_WIDTH-1 downto 0);
    signal ROM : ${name}_array_t:=(${rom_value});
    attribute rom_style : string;
    attribute rom_style of ROM : signal is "${resource_option}";
begin
    ROM_process: process(addr)
    begin
        if (en = '1') then
            data <= ROM(to_integer(unsigned(addr)));
        end if;
    end process ROM_process;
end architecture rtl;
