library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;

entity ${name} is
    generic (
        RAM_WIDTH : integer := 64;
        RAM_DEPTH_WIDTH : integer := 8;
        RAM_PERFORMANCE : string := "LOW_LATENCY";
        RESOURCE_OPTION : string := "auto";
        INIT_FILE : string := ""
    );
    port (
        addra : in std_logic_vector((RAM_DEPTH_WIDTH-1) downto 0);
        addrb : in std_logic_vector((RAM_DEPTH_WIDTH-1) downto 0);
        dina  : in std_logic_vector(RAM_WIDTH-1 downto 0);
        clka  : in std_logic;
        clkb  : in std_logic;
        wea   : in std_logic;
        enb   : in std_logic;
        rstb  : in std_logic;
        regceb: in std_logic;
        doutb : out std_logic_vector(RAM_WIDTH-1 downto 0)
    );
end ${name};

architecture rtl of ${name} is
    constant C_RAM_WIDTH : integer := RAM_WIDTH;
    constant C_RAM_DEPTH : integer := 2**RAM_DEPTH_WIDTH;
    constant C_RAM_PERFORMANCE : string := RAM_PERFORMANCE;
    constant C_INIT_FILE : string := INIT_FILE;

    signal doutb_reg : std_logic_vector(C_RAM_WIDTH-1 downto 0) := (others => '0');
    type ram_type is array (0 to C_RAM_DEPTH-1) of std_logic_vector(C_RAM_WIDTH-1 downto 0);
    signal ram_data : std_logic_vector(C_RAM_WIDTH-1 downto 0);

    function init_from_file_or_zeroes(ramfile : string) return ram_type is
    begin
        return (others => (others => '0'));
    end;

    signal ram_name : ram_type := init_from_file_or_zeroes(C_INIT_FILE);
    attribute ram_style : string;
    attribute ram_style of ram_name : signal is RESOURCE_OPTION;

begin

    process(clka)
    begin
        if rising_edge(clka) then
            if wea = '1' then
                ram_name(to_integer(unsigned(addra))) <= dina;
            end if;
        end if;
    end process;

    process(clkb)
    begin
        if rising_edge(clkb) then
            if enb = '1' then
                ram_data <= ram_name(to_integer(unsigned(addrb)));
            end if;
        end if;
    end process;

    no_output_register : if C_RAM_PERFORMANCE = "LOW_LATENCY" generate
        doutb <= ram_data;
    end generate;

    output_register : if C_RAM_PERFORMANCE = "HIGH_PERFORMANCE" generate
        process(clkb)
        begin
            if rising_edge(clkb) then
                if rstb = '1' then
                    doutb_reg <= (others => '0');
                elsif regceb = '1' then
                    doutb_reg <= ram_data;
                end if;
            end if;
        end process;
        doutb <= doutb_reg;
    end generate;
end architecture rtl;
