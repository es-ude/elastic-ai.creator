library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity skeleton_input_adapter is
    generic (
        DATA_IN_WIDTH : integer;
        DATA_IN_DEPTH : integer
    );

    port (
        signal clk : in std_logic;
        signal rst : in std_logic;
        signal start : in std_logic;
        signal input_wr_enable : in std_logic;
        signal input_wr_address : in std_logic_vector(15 downto 0);
        signal input_wr_data : in std_logic_vector(7 downto 0);
        signal stream_valid : out std_logic;
        signal stream_data : out std_logic_vector(DATA_IN_WIDTH - 1 downto 0);
        signal stream_done : out std_logic
    );
end entity;

architecture rtl of skeleton_input_adapter is
    constant IN_WORD_WIDTH : natural := get_width_in_bytes(DATA_IN_WIDTH) * 8;
    constant IN_NUMS : natural := DATA_IN_DEPTH * get_width_in_bytes(DATA_IN_WIDTH);
    constant IN_WORD_ADDR_WIDTH : natural := log2(fmax(DATA_IN_DEPTH, 2));
    constant IN_ADDR_WIDTH : natural := log2(fmax(IN_NUMS, 2));

    signal input_bram_read_enable : std_logic := '0';
    signal input_bram_read_address : std_logic_vector(IN_WORD_ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal input_bram_read_data : std_logic_vector(IN_WORD_WIDTH - 1 downto 0);
    signal input_bram_read_valid : std_logic;
    signal input_bram_write_address : std_logic_vector(IN_ADDR_WIDTH - 1 downto 0) := (others => '0');
    signal reader_stream_valid : std_logic;
    signal reader_stream_data : std_logic_vector(IN_WORD_WIDTH - 1 downto 0);
begin
    input_bram_write_address <=
        std_logic_vector(resize(unsigned(input_wr_address), IN_ADDR_WIDTH));
    stream_valid <= reader_stream_valid;
    stream_data <= reader_stream_data(DATA_IN_WIDTH - 1 downto 0);

    input_bram_i : entity work.asymmetric_dual_port_bram(rtl)
        generic map (
            WRITE_DATA_WIDTH => 8,
            WRITE_ADDRESS_WIDTH => IN_ADDR_WIDTH,
            WRITE_SIZE => IN_NUMS,
            READ_DATA_WIDTH => IN_WORD_WIDTH,
            READ_ADDRESS_WIDTH => IN_WORD_ADDR_WIDTH,
            READ_SIZE => DATA_IN_DEPTH
        )
        port map (
            read_clk => clk,
            read_address => input_bram_read_address,
            read_enable => input_bram_read_enable,
            d_out => input_bram_read_data,
            d_out_valid => input_bram_read_valid,
            write_clk => clk,
            write_address => input_bram_write_address,
            write_enable => input_wr_enable,
            d_in => input_wr_data
        );

    input_reader_i : entity work.buffered_network_input_reader(rtl)
        generic map (
            DATA_DEPTH => DATA_IN_DEPTH,
            ADDR_WIDTH => IN_WORD_ADDR_WIDTH,
            WORD_WIDTH => IN_WORD_WIDTH
        )
        port map (
            clk => clk,
            rst => rst,
            start => start,
            bram_read_valid => input_bram_read_valid,
            bram_read_data => input_bram_read_data,
            bram_read_enable => input_bram_read_enable,
            bram_read_address => input_bram_read_address,
            stream_valid => reader_stream_valid,
            stream_data => reader_stream_data,
            done => stream_done
        );
end architecture;
