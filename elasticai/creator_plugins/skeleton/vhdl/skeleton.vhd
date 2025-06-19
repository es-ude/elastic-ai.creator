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
        DATA_OUT_DEPTH : integer;
        NETWORK_IN_SIZE : integer;
        STRIDE: integer
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
    type control_t is record
        valid : std_logic;
        ready : std_logic;
    end record;
    subtype byte_t is std_logic_vector(7 downto 0);
    subtype ai_in_t is std_logic_vector(NETWORK_IN_SIZE*DATA_IN_WIDTH - 1 downto 0);
    constant ENABLE_ADDRESS : natural := 16;
    constant RESERVED_ADDRESS : natural := 17;
    constant DATA_IN_SEGMENT_START : natural := 18;
    constant DATA_OUT_SEGMENT_START : natural := DATA_IN_SEGMENT_START;
    constant DATA_IN_WIDTH_AS_BYTES : natural := size_in_bytes(DATA_IN_WIDTH);
    constant DATA_OUT_WIDTH_AS_BYTES : natural := size_in_bytes(DATA_OUT_WIDTH);
    constant DATA_IN_SEGMENT_END : natural := DATA_IN_SEGMENT_START + DATA_IN_WIDTH_AS_BYTES*DATA_IN_DEPTH - 1;
    constant DATA_OUT_SEGMENT_END : natural := DATA_OUT_SEGMENT_START + DATA_OUT_WIDTH_AS_BYTES * DATA_OUT_DEPTH - 1;


    signal network_address : std_logic_vector(15 downto 0);
    signal network_address_i : integer range 0 to 2**16 - 1;
    signal network_enable : std_logic := '0';
    signal network_d_out : STD_LOGIC_VECTOR(8 - 1 downto 0);
    signal done :  std_logic;
    signal write_data : std_logic;
    signal read_data : std_logic;

    constant skeleton_id_str : skeleton_id_t := SKELETON_ID;
    signal address_in_i : integer range 0 to 2000 := 0;
    signal network_is_enabled : std_logic;
    signal read_skeleton : std_logic;

    function addressing_data_in(signal address: in integer) return boolean is
    begin
        return address >= DATA_IN_SEGMENT_START and address <= DATA_IN_SEGMENT_END;
    end function;
    function addressing_data_out(signal address: in integer) return boolean is
    begin
        return address >= DATA_OUT_SEGMENT_START and address <= DATA_OUT_SEGMENT_END;
    end function;
begin

    address_in_i <= to_integer(unsigned(address_in));

    unpadder : entity work.dynamic_padding_remover(rtl)
    generic map (
        DATA_WIDTH => DATA_IN_WIDTH
    )
    port map (
        clk => clock,
        rst => reset,
        valid_in => write_data,
        ready_in => in_buffer_out.ready,
        d_in => data_in,
        valid_out => unpadder_out.valid,
        ready_out => unpadder_out.ready,
        d_out => unpadder_d_out
    );

    write_data <= '1' when addressing_data_in(address_in_i) and wr = '1' else '0';

    in_buffer : entity work.full_window_read_buffer(rtl)
    generic map (
        DATA_WIDTH => DATA_IN_WIDTH,
        DATA_DEPTH => DATA_IN_DEPTH,
        OUTPUT_WIDTH => NETWORK_IN_SIZE,
        STRIDE => STRIDE
    )
    port map (
        clk => clock,
        rst => reset,
        valid_in => unpadder_out.valid,
        valid_out => in_buffer_out.valid,
        ready_in => network_is_enabled,
        ready_out => in_buffer_out.ready,
        d_in => unpadder_d_out,
        d_out => in_buffer_d_out
    );

    enable_network:
    process (clock) is begin
        if rising_edge(clock) then
            if address_in_i = ENABLE_ADDRESS then
                network_enable <= data_in(0);
            end if;
        end if;
    end process;

    network_is_enabled <= network_enable and in_buffer_out.valid and not done;

    network_i: entity work.network(rtl)
    port map (
        clk => clock,
        valid_in => network_is_enabled,
        d_in => in_buffer_d_out,
        d_out => network_d_out,
        rst => reset,
        valid_out => network_valid_out
    );

    out_buffer: entity work.fifo_buffer(rtl)
    generic map (
        DATA_WIDTH => DATA_OUT_WIDTH,
        DATA_DEPTH => DATA_OUT_DEPTH
    )
    port map (
        clk => clock,
        rst => reset,
        d_in => network_d_out,
        d_out => out_buffer_d_out,
        valid_in => network_valid_out,
        ready_in => padder_out.ready,
        valid_out => out_buffer_out.valid,
        ready_out => out_buffer_out.ready
    );

    padder : entity work.dynamic_padder(rtl)
    generic map (
        DATA_WIDTH => DATA_OUT_WIDTH
    )
    port map (
        clk => clock,
        rst => reset,
        d_in => out_buffer_d_out,
        d_out => padder_d_out,
        valid_in => out_buffer_out.valid,
        valid_out => padder_out.valid,
        ready_in => read_data,
        ready_out => padder_out.ready
    );

    unpadder_in.valid <= '1' when addressing_data_in(address_in_i) else '0';
    data_out <=  skeleton_id_str(address_in_i) when read_skeleton = '1' else padder_d_out;

    read_data <= '1' when rd = '1' and addressing_data_out(address_in_i);
    read_skeleton <= '1' when rd = '1' and address_in_i < 16 else '0';

    busy <= not done;
    wake_up <= done;


    check_done:
    process (clock, reset) is
    begin
        if reset = '1' then
            done <= '0';
        elsif rising_edge(clock) then
            -- out buffer is full
            if out_buffer_out.ready = '0' then
                done <= '1'; 
            end if;
        end if;
    end process;




end rtl;
