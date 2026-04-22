library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

-- Top-level middleware adapter for the skeleton hardware.
-- It removes the address offsets introduced by the skeleton/hw_function_id
-- layout, decodes byte reads and writes, and forwards control/data flow to the
-- inference controller.

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
    constant DATA_IN_WIDTH_AS_BYTES : natural := get_width_in_bytes(DATA_IN_WIDTH);
    constant DATA_OUT_WIDTH_AS_BYTES : natural := get_width_in_bytes(DATA_OUT_WIDTH);

    signal network_enable :  std_logic := '0';
    signal rst_network : std_logic;

    signal done :  std_logic := '0';

    signal skeleton_id_str : skeleton_id_t := SKELETON_ID;
    signal in_wr_enable : std_logic := '0';
    signal in_wr_address : std_logic_vector(15 downto 0) := (others => '0');
    signal in_wr_data : std_logic_vector(7 downto 0) := (others => '0');
    signal out_rd_enable : std_logic := '0';
    signal out_rd_address : std_logic_vector(15 downto 0) := (others => '0');
    signal out_rd_data : std_logic_vector(7 downto 0) := (others => '0');
    signal out_data_pending : std_logic := '0';
    signal address_in_i : integer range 0 to 2000 := 0;

begin
    address_in_i <= to_integer(unsigned(address_in));

    rst_network <= reset;

    rx_tx_middleware: process (clock)
    begin
        if rising_edge(clock) then
            in_wr_enable <= '0';
            out_rd_enable <= '0';

            if out_data_pending = '1' then
                data_out <= out_rd_data;
                out_data_pending <= '0';
            end if;

            if reset = '1' then
                network_enable <= '0';
                data_out <= (others => '0');
                out_data_pending <= '0';
            else
                if wr = '1' then
                    if address_in_i = ENABLE_ADDRESS then
                        network_enable <= data_in(0);
                    elsif address_in_i >= DATA_IN_SEGMENT_START and address_in_i <= DATA_IN_SEGMENT_START + DATA_IN_WIDTH_AS_BYTES*DATA_IN_DEPTH - 1 then
                        in_wr_enable <= '1';
                        in_wr_address <= std_logic_vector(
                            to_unsigned(address_in_i - DATA_IN_SEGMENT_START, in_wr_address'length)
                        );
                        in_wr_data <= data_in;
                    end if;
                end if;
                if rd = '1' then
                    if address_in_i >= 0 and address_in_i <= 15 then
                        data_out <= skeleton_id_str(address_in_i);
                    elsif address_in_i >= DATA_OUT_SEGMENT_START and address_in_i <= DATA_OUT_SEGMENT_START + DATA_OUT_WIDTH_AS_BYTES*DATA_OUT_DEPTH - 1 then
                        out_rd_enable <= '1';
                        out_rd_address <= std_logic_vector(
                            to_unsigned(address_in_i - DATA_OUT_SEGMENT_START, out_rd_address'length)
                        );
                        out_data_pending <= '1';
                    end if;
                end if;
            end if;
        end if;
    end process;

    network_i : entity work.skeleton_inference_controller(rtl)
        generic map (
            DATA_IN_WIDTH => DATA_IN_WIDTH,
            DATA_IN_DEPTH => DATA_IN_DEPTH,
            DATA_OUT_WIDTH => DATA_OUT_WIDTH,
            DATA_OUT_DEPTH => DATA_OUT_DEPTH
        )
        port map (
            clk => clock,
            rst => rst_network,
            done => done,
            network_enable => network_enable,
            input_wr_enable => in_wr_enable,
            input_wr_address => in_wr_address,
            input_wr_data => in_wr_data,
            output_rd_enable => out_rd_enable,
            output_rd_address => out_rd_address,
            output_rd_data => out_rd_data
        );



    busy <= not done;
    wake_up <= done;


end rtl;
