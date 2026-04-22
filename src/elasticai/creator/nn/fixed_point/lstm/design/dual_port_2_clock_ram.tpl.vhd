--  based on xilinx_simple_dual_port_2_clock_ram
--  but we did some custom modifications(chao)
--  Xilinx Simple Dual Port 2 Clock RAM
--  This code implements a parameterizable SDP dual clock memory.
--  If a reset or enable is not necessary, it may be tied off or removed from the code.

library ieee;
use ieee.std_logic_1164.all;


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

USE std.textio.all;

entity dual_port_2_clock_ram_${name} is
generic (
    RAM_WIDTH : integer := 64;                      -- Specify RAM data width
    RAM_DEPTH_WIDTH : integer := 8;                    -- Specify RAM depth (number of entries)
    RAM_PERFORMANCE : string := "LOW_LATENCY";      -- Select "HIGH_PERFORMANCE" or "LOW_LATENCY"
    INIT_FILE : string := ""                        -- Specify name/location of RAM initialization file if using one (leave blank if not)
    );

port (
        addra : in std_logic_vector((RAM_DEPTH_WIDTH-1) downto 0);     -- Write address bus, width determined from RAM_DEPTH
        addrb : in std_logic_vector((RAM_DEPTH_WIDTH-1) downto 0);     -- Read address bus, width determined from RAM_DEPTH
        dina  : in std_logic_vector(RAM_WIDTH-1 downto 0);		  -- RAM input data
        clka  : in std_logic;                       			  -- Write Clock
        clkb  : in std_logic;                       			  -- Read Clock
        wea   : in std_logic;                       			  -- Write enable
        enb   : in std_logic;                       			  -- RAM Enable, for additional power savings, disable port when not in use
        rstb  : in std_logic;                       			  -- Output reset (does not affect memory contents)
        regceb: in std_logic;                       			  -- Output register enable
        doutb : out std_logic_vector(RAM_WIDTH-1 downto 0)   			  -- RAM output data
    );

end dual_port_2_clock_ram_${name};

architecture rtl of dual_port_2_clock_ram_${name} is

constant C_RAM_WIDTH : integer := RAM_WIDTH;
constant C_RAM_DEPTH : integer := 2**RAM_DEPTH_WIDTH;
constant C_RAM_PERFORMANCE : string := RAM_PERFORMANCE;
constant C_INIT_FILE : string := INIT_FILE;


signal doutb_reg : std_logic_vector(C_RAM_WIDTH-1 downto 0) := (others => '0');

type ram_type is array (0 to C_RAM_DEPTH-1) of std_logic_vector (C_RAM_WIDTH-1 downto 0);          -- 2D Array Declaration for RAM signal

signal ram_data : std_logic_vector(C_RAM_WIDTH-1 downto 0) ;


function init_from_file_or_zeroes(ramfile : string) return ram_type is
begin
--    if ramfile = "" then --if the file name is empty then init ram with 0
    return (others => (others => '0'));
--    else
--        return InitRamFromFile(ramfile) ;
--    end if;
end;
-- Following code defines RAM

signal ram_name : ram_type := init_from_file_or_zeroes(C_INIT_FILE);

begin

process(clka)
begin
    if(clka'event and clka = '1') then
        if(wea = '1') then
            ram_name(to_integer(unsigned(addra))) <= dina;
        end if;
    end if;
end process;

process(clkb)
begin
    if(clkb'event and clkb = '1') then
        if(enb = '1') then
            ram_data <= ram_name(to_integer(unsigned(addrb)));
        end if;
    end if;
end process;


--  Following code generates LOW_LATENCY (no output register)
--  Following is a 1 clock cycle read latency at the cost of a longer clock-to-out timing

no_output_register : if C_RAM_PERFORMANCE = "LOW_LATENCY" generate
    doutb <= ram_data;
end generate;

--  Following code generates HIGH_PERFORMANCE (use output register)
--  Following is a 2 clock cycle read latency with improved clock-to-out timing

output_register : if C_RAM_PERFORMANCE = "HIGH_PERFORMANCE"  generate
    process(clkb)
    begin
        if(clkb'event and clkb = '1') then
            if(rstb = '1') then
                doutb_reg <= (others => '0');
            elsif(regceb = '1') then
                doutb_reg <= ram_data;
            end if;
        end if;
    end process;
    doutb <= doutb_reg;
end generate;

end rtl;
