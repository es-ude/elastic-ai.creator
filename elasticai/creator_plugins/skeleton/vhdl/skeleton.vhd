library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.padding_pkg.all;
use work.skeleton_pkg.all;

entity skeleton is
    generic (
        DATA_IN_WIDTH : integer;
        DATA_IN_DEPTH : integer;
        DATA_OUT_WIDTH : integer;
        DATA_OUT_DEPTH : integer
    );
    port (
        -- control interface
        clock                : in std_logic;
        clk_hadamard                : in std_logic;
        reset                : in std_logic; -- controls functionality (sleep)
        busy                : out std_logic; -- done with entire calculation
        wake_up             : out std_logic;
        -- indicate new data or request
        rd                    : in std_logic;    -- request a variable
        wr                 : in std_logic;     -- request changing a variable

        -- data interface
        data_in            : in std_logic_vector(7 downto 0);
        address_in        : in std_logic_vector(15 downto 0);
        data_out            : out std_logic_vector(7 downto 0);

        debug                : out std_logic_vector(7 downto 0);

        led_ctrl             : out std_logic_vector(3 DOWNTO 0)
    );
end;

architecture rtl of skeleton is
    constant ENABLE_ADDRESS : natural := 16;
    constant RESERVED_ADDRESS : natural := 17;
    constant DATA_IN_SEGMENT_START : natural := 18;
    constant DATA_OUT_SEGMENT_START : natural := DATA_IN_SEGMENT_START;
    constant DATA_IN_WIDTH_AS_BYTES : natural := size_in_bytes(DATA_IN_WIDTH);
    constant DATA_OUT_WIDTH_AS_BYTES : natural := size_in_bytes(DATA_OUT_WIDTH);

    signal network_enable :  std_logic;
    signal rst_network : std_logic;

    signal done :  std_logic;

    signal skeleton_id_str : skeleton_id_t := SKELETON_ID;
     signal mw_rx_buffer_bytes : byte_array_t(DATA_IN_DEPTH * DATA_IN_WIDTH_AS_BYTES + DATA_IN_SEGMENT_START - 1
                                        downto DATA_IN_SEGMENT_START);
    signal mw_rx_buffer_bytes_without_offset : byte_array_t(DATA_IN_DEPTH * DATA_IN_WIDTH_AS_BYTES - 1
                                                            downto 0);
    signal mw_rx_buffer_v : std_logic_vector(mw_rx_buffer_bytes'length * 8 - 1 downto 0);
    signal mw_tx_buffer_bytes : byte_array_t(DATA_OUT_DEPTH * DATA_OUT_WIDTH_AS_BYTES - 1 + DATA_OUT_SEGMENT_START
                                        downto DATA_OUT_SEGMENT_START);
    signal mw_tx_buffer_bytes_without_offset : byte_array_t(DATA_OUT_DEPTH * DATA_OUT_WIDTH_AS_BYTES - 1 downto 0);
    signal mw_tx_buffer_v : std_logic_vector(mw_tx_buffer_bytes'length * 8 - 1 downto 0);
    signal address_in_i : integer range 0 to 2000 := 0;

begin

     mw_rx_buffer_bytes_without_offset <= mw_rx_buffer_bytes;
    mw_tx_buffer_bytes <= mw_tx_buffer_bytes_without_offset;
    address_in_i <= to_integer(unsigned(address_in));



    flatten_mw_rx_buffer:
    for byte_id in 0 to DATA_IN_DEPTH*DATA_IN_WIDTH_AS_BYTES - 1 generate
        mw_rx_buffer_v((byte_id + 1) * 8 - 1 downto byte_id * 8) <= mw_rx_buffer_bytes_without_offset(byte_id);
    end generate;

    flatten_mw_tx_buffer:
    for byte_id in 0 to DATA_OUT_DEPTH*DATA_OUT_WIDTH_AS_BYTES - 1 generate
        mw_tx_buffer_bytes_without_offset(byte_id) <=
            mw_tx_buffer_v((byte_id + 1) * 8 - 1 downto byte_id * 8);
    end generate;

    rst_network <= reset;

    rx_tx_middleware: process (clock)
    begin
        if rising_edge(clock) then

            if reset = '1' then
            else
                if wr = '1' then
                    if address_in_i = ENABLE_ADDRESS then
                        network_enable <= data_in(0);
                    elsif address_in_i >= DATA_IN_SEGMENT_START and address_in_i <= DATA_IN_SEGMENT_START + DATA_IN_WIDTH_AS_BYTES*DATA_IN_DEPTH - 1 then
                        mw_rx_buffer_bytes(address_in_i ) <= data_in;
                    end if;
                end if;
                if rd = '1' then
                    if address_in_i >= 0 and address_in_i <= 15 then
                        data_out <= skeleton_id_str(address_in_i);
                    elsif address_in_i >= DATA_OUT_SEGMENT_START and address_in_i <= DATA_OUT_SEGMENT_START + DATA_IN_WIDTH_AS_BYTES*DATA_OUT_DEPTH - 1 then
                        data_out <= mw_tx_buffer_bytes(address_in_i);
                    end if;
                end if;
            end if;
        end if;
    end process;

    network_i : entity work.buffered_network_wrapper(rtl)
        generic map (
            DATA_IN_WIDTH => DATA_IN_WIDTH,
            DATA_IN_DEPTH => DATA_IN_DEPTH,
            DATA_OUT_WIDTH => DATA_OUT_WIDTH,
            DATA_OUT_DEPTH => DATA_OUT_DEPTH
        )
        port map (
            clk => clock,
            rst => rst_network,
            valid_out => done,
            valid_in => network_enable,
            d_in => mw_rx_buffer_v,
            d_out => mw_tx_buffer_v
        );



    busy <= not done;
    wake_up <= done;


end rtl;
