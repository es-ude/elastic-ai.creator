library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity base_shift_register is
    generic (
        DATA_WIDTH: positive;
        NUM_POINTS: positive
    );
    port (
        CLK : in std_logic;
        D_IN : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        D_OUT : out std_logic_vector(DATA_WIDTH * NUM_POINTS - 1 downto 0);
        SRC_VALID : in std_logic;
        RST : in std_logic;
        VALID : out std_logic := '0';
        READY : out std_logic := '1';
        DST_READY : in std_logic := '1';
        EN : in std_logic := '1'
    );
end entity;

architecture rtl of base_shift_register is
    signal storage : std_logic_vector(DATA_WIDTH*NUM_POINTS - 1 downto 0) := (others => '0');
  


begin
    
    READY <= DST_READY;  -- Ready when downstream is ready (simple passthrough)
    D_OUT <= storage;

    process(CLK) is
        variable counter : integer := 0;
    begin
        if rising_edge(CLK) then
            VALID <= '0';
            if RST = '1' then
                counter := 0;

            elsif SRC_VALID = '1' and DST_READY = '1' and EN = '1' then
                    storage <= storage((DATA_WIDTH*(NUM_POINTS-1))-1 downto 0) & d_in;
                    if counter < NUM_POINTS - 1 then
                        counter := counter + 1;
                    else
                        VALID <= '1';
                    end if;
                end if;
            end if;
    end process;
end architecture;
