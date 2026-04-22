library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity skeleton_network_runner is
    generic (
        DATA_IN_WIDTH : integer;
        DATA_OUT_WIDTH : integer;
        DATA_OUT_DEPTH : integer
    );

    port (
        signal clk : in std_logic;
        signal rst : in std_logic;
        signal start : in std_logic;
        signal stream_valid : in std_logic;
        signal stream_data : in std_logic_vector(DATA_IN_WIDTH - 1 downto 0);
        signal output_word_valid : out std_logic;
        signal output_word_address : out std_logic_vector(log2(fmax(DATA_OUT_DEPTH, 2)) - 1 downto 0);
        signal output_word_data : out std_logic_vector(get_width_in_bytes(DATA_OUT_WIDTH) * 8 - 1 downto 0);
        signal done : out std_logic
    );
end entity;

architecture rtl of skeleton_network_runner is
    constant OUT_WORD_WIDTH : natural := get_width_in_bytes(DATA_OUT_WIDTH) * 8;
    constant OUT_WORD_ADDR_WIDTH : natural := log2(fmax(DATA_OUT_DEPTH, 2));

    signal active : std_logic := '0';
    signal network_valid_in : std_logic := '0';
    signal network_valid_out : std_logic;
    signal network_ready : std_logic;
    signal network_data_in : std_logic_vector(DATA_IN_WIDTH - 1 downto 0) := (others => '0');
    signal network_data_out : std_logic_vector(DATA_OUT_WIDTH - 1 downto 0);
    signal output_idx : integer range 0 to DATA_OUT_DEPTH := 0;
begin
    network_i : entity work.network(rtl)
        port map (
            CLK => clk,
            D_IN => network_data_in,
            D_OUT => network_data_out,
            SRC_VALID => network_valid_in,
            RST => rst,
            VALID => network_valid_out,
            READY => network_ready,
            DST_READY => '1',
            EN => '1'
        );

    process(clk)
        variable out_word : std_logic_vector(OUT_WORD_WIDTH - 1 downto 0);
    begin
        if rising_edge(clk) then
            network_valid_in <= '0';
            output_word_valid <= '0';
            done <= '0';

            if rst = '1' then
                active <= '0';
                output_idx <= 0;
                output_word_address <= (others => '0');
                output_word_data <= (others => '0');
            else
                if start = '1' then
                    active <= '1';
                    output_idx <= 0;
                end if;

                if active = '1' and stream_valid = '1' then
                    network_valid_in <= '1';
                    network_data_in <= stream_data;
                end if;

                if network_valid_out = '1' and output_idx < DATA_OUT_DEPTH then
                    out_word := (others => '0');
                    out_word(DATA_OUT_WIDTH - 1 downto 0) := network_data_out;
                    output_word_valid <= '1';
                    output_word_address <= std_logic_vector(to_unsigned(output_idx, OUT_WORD_ADDR_WIDTH));
                    output_word_data <= out_word;

                    if output_idx = DATA_OUT_DEPTH - 1 then
                        done <= '1';
                        active <= '0';
                    else
                        output_idx <= output_idx + 1;
                    end if;
                end if;
            end if;
        end if;
    end process;
end architecture;
