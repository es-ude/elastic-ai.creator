library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

-- Reads data from BRAM and feeds it to 
-- a neural network using valid/ready
-- handshake signals.

entity buffered_network_input_reader is
    generic (
        DATA_DEPTH : integer;
        ADDR_WIDTH : integer;
        WORD_WIDTH : integer
    );

    port (
        signal clk : in std_logic;
        signal rst : in std_logic;
        signal start : in std_logic;
        signal bram_read_valid : in std_logic;
        signal bram_read_data : in std_logic_vector(WORD_WIDTH - 1 downto 0);
        signal bram_read_enable : out std_logic;
        signal bram_read_address : out std_logic_vector(ADDR_WIDTH - 1 downto 0);
        signal stream_valid : out std_logic;
        signal stream_data : out std_logic_vector(WORD_WIDTH - 1 downto 0);
        signal done : out std_logic
    );
end entity;

architecture rtl of buffered_network_input_reader is
    signal active : std_logic := '0';
    signal issued_reads : integer range 0 to DATA_DEPTH := 0;
    signal consumed_reads : integer range 0 to DATA_DEPTH := 0;
    signal bram_read_address_i : std_logic_vector(ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal stream_data_i : std_logic_vector(WORD_WIDTH - 1 downto 0) := (others => '0');
begin
    bram_read_address <= bram_read_address_i;
    stream_data <= stream_data_i;

    process(clk)
    begin
        if rising_edge(clk) then
            bram_read_enable <= '0';
            stream_valid <= '0';
            done <= '0';

            if rst = '1' then
                active <= '0';
                issued_reads <= 0;
                consumed_reads <= 0;
                bram_read_address_i <= (others => '0');
                stream_data_i <= (others => '0');
            else
                if start = '1' and active = '0' then
                    active <= '1';
                    issued_reads <= 0;
                    consumed_reads <= 0;
                end if;

                if active = '1' then
                    if issued_reads < DATA_DEPTH then
                        bram_read_enable <= '1';
                        bram_read_address_i <= std_logic_vector(to_unsigned(issued_reads, ADDR_WIDTH));
                        issued_reads <= issued_reads + 1;
                    end if;

                    if bram_read_valid = '1' and consumed_reads < DATA_DEPTH then
                        stream_valid <= '1';
                        stream_data_i <= bram_read_data;
                        consumed_reads <= consumed_reads + 1;
                    end if;

                    if issued_reads = DATA_DEPTH and consumed_reads = DATA_DEPTH then
                        done <= '1';
                        active <= '0';
                    end if;
                end if;
            end if;
        end if;
    end process;
end architecture;
