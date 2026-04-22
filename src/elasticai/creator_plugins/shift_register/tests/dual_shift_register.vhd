library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dual_shift_register is
    generic (
        DATA_WIDTH: positive := 8;
        NUM_POINTS_1: positive := 3;
        NUM_POINTS_2: positive := 2
    );
    port (
        CLK : in std_logic;
        D_IN : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        D_OUT : out std_logic_vector(DATA_WIDTH *  NUM_POINTS_2 - 1 downto 0);
        SRC_VALID : in std_logic;
        RST : in std_logic;
        VALID : out std_logic := '0';
        READY : out std_logic := '1';
        DST_READY : in std_logic := '1';
        EN : in std_logic := '1'
    );
end entity;

architecture rtl of dual_shift_register is


    -- Intermediate signals between the two shift registers
    signal sr1_out : std_logic_vector(DATA_WIDTH * NUM_POINTS_1 - 1 downto 0);
    signal sr1_valid : std_logic;
    signal sr1_ready : std_logic;
    signal sr2_ready : std_logic;
    signal sr2_in : std_logic_vector(DATA_WIDTH - 1 downto 0);
    
    function or_reduce(vec : std_logic_vector) return std_logic_vector is
        variable result : std_logic_vector(DATA_WIDTH - 1 downto 0) := (others => '0');
    begin
        for i in 0 to (NUM_POINTS_1 - 1) loop
            result := result or vec(DATA_WIDTH*(i+1) - 1 downto DATA_WIDTH*i);
        end loop;
        return result;

    end function;

begin

    sr2_in <= or_reduce(sr1_out);
    -- First shift register: takes scalar input, produces NUM_POINTS_1 window
    shift_reg_1: entity work.base_shift_register(rtl)
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            NUM_POINTS => NUM_POINTS_1
        )
        port map (
            CLK => CLK,
            D_IN => D_IN,
            D_OUT => sr1_out,
            SRC_VALID => SRC_VALID,
            RST => RST,
            VALID => sr1_valid,
            READY => sr1_ready,
            DST_READY => sr2_ready,
            EN => EN
        );
    
    

    -- Second shift register: takes NUM_POINTS_1 window, produces NUM_POINTS_1 * NUM_POINTS_2 output
    -- This creates a sliding window of windows
    shift_reg_2: entity work.base_shift_register(rtl)
        generic map (
            DATA_WIDTH => DATA_WIDTH,
            NUM_POINTS => NUM_POINTS_2
        )
        port map (
            CLK => CLK,
            D_IN => sr2_in,
            D_OUT => D_OUT,
            SRC_VALID => sr1_valid,
            RST => RST,
            VALID => VALID,
            READY => sr2_ready,
            DST_READY => DST_READY,
            EN => EN
        );

    -- Backpressure: propagate sr2's ready signal to sr1
    READY <= sr1_ready;

end architecture;
