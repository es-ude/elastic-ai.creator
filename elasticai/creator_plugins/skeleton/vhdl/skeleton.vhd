library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity skeleton is
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
    constant DATA_SEGMENT_START : natural := 18;
    constant skeleton_id_str : skeleton_id_t := SKELETON_ID;
    signal skeleton_address_i : integer range skeleton_id_t'range;
    signal network_address : std_logic_vector(15 downto 0);
    signal network_address_i : integer range 0 to 2**16 - 1;
    signal network_enable : std_logic := '0';
    signal network_d_out : STD_LOGIC_VECTOR(8 - 1 downto 0);
    signal done :  std_logic;
    signal network_wr : std_logic;
    signal address_in_i : integer range 0 to 2**16 - 1 := 0;

    type read_state_t is (rd_skeleton_id, rd_network, rd_nothing);
    type write_state_t is (wr_enable, wr_network, wr_nothing);

    signal read_state : read_state_t := rd_nothing;
    signal write_state : write_state_t := wr_nothing;
    signal current_skeleton_id_byte : std_logic_vector(7 downto 0);
begin

    address_in_i <= TO_INTEGER(UNSIGNED(address_in));
    network_address <= std_logic_vector(to_unsigned(network_address_i, network_address'length));
    
    network_address_i <= max_fn(address_in_i - DATA_SEGMENT_START, 0);
    network_wr <= '1' when write_state = wr_network else '0';
    skeleton_address_i <= min_fn(address_in_i, skeleton_id_t'high);
    current_skeleton_id_byte <= skeleton_id_str(skeleton_address_i) when read_state = rd_skeleton_id else (others => 'X');

    update_write_state:
    process (wr, address_in_i) is
    begin
        if wr = '1' then
            if address_in_i >= DATA_SEGMENT_START then
                write_state <= wr_network;
            elsif address_in_i = ENABLE_ADDRESS then
                write_state <= wr_enable;
            else
                write_state <= wr_nothing;
            end if;
        else
            write_state <= wr_nothing;
        end if;
    end process;

    update_read_state:
    process (rd, address_in_i, wr) is
    begin
        if wr = '0' and rd = '1' then
            if address_in_i >= DATA_SEGMENT_START then
                read_state <= rd_network;
            elsif address_in_i < ENABLE_ADDRESS then
                read_state <= rd_skeleton_id;
            else
                read_state <= rd_nothing;
            end if;
        else
            read_state <= rd_nothing;
        end if;
    end process;
        

    update_enable:
    process (clock) is
    begin
        if rising_edge(clock) then
            if write_state = wr_enable then
                network_enable <= data_in(0);
            end if;
        end if;
    end process;

    
    update_data_out:
    process (read_state, current_skeleton_id_byte, network_d_out) is
    begin
        case read_state is
            when rd_skeleton_id => data_out <= current_skeleton_id_byte;
            when rd_network => data_out <= network_d_out;
            when others => data_out <= (others => 'Z');
        end case;
    end process;



    network : entity work.buffered_network_wrapper(rtl)
        port map(
            d_in => data_in,
            d_out => network_d_out,
            address => network_address,
            done => done,
            wr => network_wr,
            enable => network_enable,
            rst => reset,
            clk => clock
        );

    busy <= not done;
    wake_up <= done;
end rtl;
