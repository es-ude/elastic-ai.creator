library ieee;
    use ieee.std_logic_1164.all;
    use ieee.std_logic_unsigned.all;
entity ${rom_name}_${layer_name} is
    port (
        clk : in std_logic;
        en : in std_logic;
        addr : in std_logic_vector(${rom_addr_bitwidth}-1 downto 0);
        data : out std_logic_vector(${rom_data_bitwidth}-1 downto 0)
    );
end entity ${rom_name}_${layer_name};
architecture rtl of ${rom_name}_${layer_name} is
    type ${rom_name}_${layer_name}_array_t is array (0 to 2**${rom_addr_bitwidth}-1) of std_logic_vector(${rom_data_bitwidth}-1 downto 0);
    signal ROM : ${rom_name}_${layer_name}_array_t:=(${rom_value});
    attribute rom_style : string;
    attribute rom_style of ROM : signal is ${rom_resource_option};
begin
    ROM_process: process(clk)
    begin
        if rising_edge(clk) then
            if (en = '1') then
                data <= ROM(conv_integer(addr));
            end if;
        end if;
    end process ROM_process;
end architecture rtl;
