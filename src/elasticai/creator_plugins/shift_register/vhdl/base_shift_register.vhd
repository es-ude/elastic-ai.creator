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
    type buffer_t is array (0 to NUM_POINTS - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal storage : buffer_t := (others => (others => '0'));
    signal write_ptr : integer range 0 to NUM_POINTS - 1 := 0;
    signal filled : integer range 0 to NUM_POINTS := 0;
    signal valid_reg : std_logic := '0';

    function build_window(buff : buffer_t; write_pos : integer; fill_count : integer) return std_logic_vector is
        variable assembled : std_logic_vector(DATA_WIDTH * NUM_POINTS - 1 downto 0) := (others => '0');
        variable current_index : integer := 0;
    begin
        if fill_count > 0 then
            current_index := write_pos - 1;
            for chunk_idx in 0 to fill_count - 1 loop
                if current_index < 0 then
                    current_index := current_index + NUM_POINTS;
                end if;
                assembled((chunk_idx + 1) * DATA_WIDTH - 1 downto chunk_idx * DATA_WIDTH) := buff(current_index);
                current_index := current_index - 1;
            end loop;
        end if;
        return assembled;
    end function;
begin
    
    VALID <= valid_reg;
    READY <= DST_READY;  -- Ready when downstream is ready (simple passthrough)
    D_OUT <= build_window(storage, write_ptr, filled);

    update_register : process(CLK, RST)
    begin
        if RST = '1' then
            storage <= (others => (others => '0'));
            write_ptr <= 0;
            filled <= 0;
            valid_reg <= '0';
        elsif rising_edge(CLK) then
            -- Default: keep valid low unless we're asserting it
            valid_reg <= '0';
            -- Level-sensitive handshake: write when both src_valid and dst_ready are high
            if SRC_VALID = '1' and DST_READY = '1' then
                storage(write_ptr) <= D_IN;
                if write_ptr = NUM_POINTS - 1 then
                    write_ptr <= 0;
                else
                    write_ptr <= write_ptr + 1;
                end if;
                if filled < NUM_POINTS then
                    filled <= filled + 1;
                    -- Assert valid when buffer becomes full (on the cycle we write the last element)
                    if filled = NUM_POINTS - 1 then
                        valid_reg <= '1';
                    end if;
                else
                    -- Buffer already full, new write creates new valid window
                    filled <= NUM_POINTS;
                    valid_reg <= '1';
                end if;
            end if;
        end if;
    end process;

end architecture;