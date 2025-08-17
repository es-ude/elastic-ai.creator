library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity addressable_input_buffer is
    generic (
        DATA_WIDTH : positive;
        DATA_DEPTH : positive;
        DATA_OUT_DEPTH : positive := 1;
        STRIDE : positive := 1
    );

    port (
        write_enable : in std_logic;
        address : in std_logic_vector(15 downto 0);
        d_in : in std_logic_vector(7 downto 0);
        d_out : out std_logic_vector(DATA_WIDTH*DATA_OUT_DEPTH - 1 downto 0);
        ready_in : in std_logic;
        valid_out : out std_logic;
        clk : in std_logic;
        rst : in std_logic
    );
end entity;

architecture rtl of addressable_input_buffer is
 
    constant DATA_WIDTH_BYTES : integer := size_in_bytes(DATA_WIDTH);
    constant TOTAL_NUM_BYTES : integer := size_in_bytes(DATA_WIDTH)*DATA_DEPTH;
    constant REQUIRED_PADDING : integer := DATA_WIDTH_BYTES*8 - DATA_WIDTH;
    constant REMAINDER : integer := 8 - REQUIRED_PADDING;
    constant TOTAL_OUT_BYTES : integer := DATA_WIDTH_BYTES * DATA_OUT_DEPTH;
    constant STRIDE_IN_BYTES : integer := DATA_WIDTH_BYTES * STRIDE;
    subtype byte_t is std_logic_vector(7 downto 0);
    type byte_array_t is array (integer range <>) of byte_t;
    subtype ram_t is byte_array_t(0 to TOTAL_NUM_BYTES - 1);
    subtype index_t is integer range 0 to ram_t'high - (TOTAL_OUT_BYTES - 1);
    signal delayed_input : std_logic_vector(7 downto 0);
    signal delayed_write_enable : std_logic;
    signal delayed_address : integer range ram_t'range := 0;
    signal ram : ram_t;
    signal read_ptr : index_t := 0;
    
    -- Add synthesis attributes for better timing
    attribute ram_style : string;
    attribute ram_style of ram : signal is "block";
    attribute use_dsp : string;
    attribute use_dsp of read_ptr : signal is "no";
    attribute max_fanout : integer;
    attribute max_fanout of read_ptr : signal is 32;

    signal address_int : integer range ram_t'range := 0;

    type unpadded_t is array (0 to DATA_OUT_DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal unpadded_d_out : unpadded_t;
    signal padded_d_out : byte_array_t(0 to DATA_OUT_DEPTH * DATA_WIDTH_BYTES - 1);
    signal valid_out_reg : std_logic := '0';


    function next_read_ptr(current: index_t; reading: std_logic) return index_t is
    begin
        if reading = '1' then
            if current + STRIDE_IN_BYTES <= index_t'high then
                return current + STRIDE_IN_BYTES;
           else
                return current; -- No more data to read
            end if;
        else
            return current; -- No change if not reading
        end if;
    end function;


    begin

    address_int <= to_integer(unsigned(address)); 

    remove_padding:
    for i in 0 to DATA_OUT_DEPTH - 1 generate
        handle_byte_slices:
        for j in 0 to DATA_WIDTH_BYTES - 1 generate
            handle_first_bytes:
            if j < DATA_WIDTH_BYTES - 1 generate
                unpadded_d_out(i)(j*8 + 7 downto j*8) <= padded_d_out(DATA_WIDTH_BYTES * i + j);
            end generate;
            handle_last_byte:
            if j = DATA_WIDTH_BYTES - 1 generate
                unpadded_d_out(i)(j*8 + REMAINDER - 1 downto j*8) <= padded_d_out(DATA_WIDTH_BYTES * i + j)(REMAINDER - 1 downto 0);
            end generate;
        end generate;
    end generate;

    connect_unpadded_d_out:
    for i in 0 to DATA_OUT_DEPTH - 1 generate
            d_out(i*DATA_WIDTH + DATA_WIDTH - 1 downto i*DATA_WIDTH) <= unpadded_d_out(DATA_OUT_DEPTH - 1 - i);
    end generate;
    
    write_data: process(clk) is
    begin
        if rising_edge(clk) then
            if delayed_write_enable = '1' then
                ram(delayed_address) <= delayed_input;
            end if;
        end if;
    end process;

    delay_signals: process(clk) is
    begin
        if rising_edge(clk) then
            delayed_input <= d_in;
            delayed_write_enable <= write_enable;
            delayed_address <= address_int;
        end if;
    end process;

    update_read_ptr: process(clk) is
    begin
        if rising_edge(clk) then
            if rst = '1' then
                read_ptr <= index_t'low;
            else
                read_ptr <= next_read_ptr(read_ptr, ready_in);
            end if;
        end if;
    end process;

    update_padded_d_out:
    
        process(clk) is
            variable next_read : index_t;
        begin
            if rising_edge(clk) then
                if ready_in = '1' and read_ptr + STRIDE_IN_BYTES <= index_t'high then
                    next_read := read_ptr + STRIDE_IN_BYTES;
                else
                    next_read := read_ptr; -- No change if not reading
                end if;
                for i in 0 to TOTAL_OUT_BYTES - 1 loop
                    padded_d_out(i) <= ram(next_read + i);
                end loop;
            end if;
        end process;

    update_valid_out: process(clk) is
    begin
        if rising_edge(clk) then
            if rst = '1' then
                valid_out_reg <= '0';
            else
                if read_ptr + STRIDE_IN_BYTES <= index_t'high then
                    valid_out_reg <= '1';
                else
                    valid_out_reg <= '0';
                end if;
            end if;
        end if;
    end process;

    valid_out <= valid_out_reg;


end architecture;
