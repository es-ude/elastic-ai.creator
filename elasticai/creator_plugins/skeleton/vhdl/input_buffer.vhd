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

    signal ram : ram_t;
    signal read_ptr : index_t := 0;
    type read_ptr_list_t is array(0 to TOTAL_OUT_BYTES - 1) of integer range 0 to DATA_WIDTH*DATA_DEPTH - 1;

    function create_read_ptr_list(index: index_t) return read_ptr_list_t is
        variable result : read_ptr_list_t := (others => 0);
    begin
        for i in read_ptr_list_t'range loop
            result(i) := index + i;
        end loop;
        return result;
    end function;

    signal read_ptr_list : read_ptr_list_t := create_read_ptr_list(0);


    signal address_int : integer range 0 to 2**16 - 1 := 0;

    type unpadded_t is array (0 to DATA_OUT_DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    signal unpadded_d_out : unpadded_t;
    signal padded_d_out : byte_array_t(0 to DATA_OUT_DEPTH * DATA_WIDTH_BYTES - 1);


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
            if write_enable = '1' then
                ram(address_int) <= d_in;
            end if;
        end if;
    end process;

    update_read_ptr: process(clk, rst) is
    begin
        if rst = '1' then
            read_ptr <= index_t'low;
        elsif rising_edge(clk) then
            read_ptr <= next_read_ptr(read_ptr, ready_in);
        end if;
    end process;

    update_read_ptr_list:
    for i in read_ptr_list_t'range generate
        read_ptr_list(i) <= read_ptr + i;
    end generate;

    update_padded_d_out:
    for i in read_ptr_list_t'range generate
        process(clk) is
        begin
            if rising_edge(clk) then
                padded_d_out(i) <= ram(next_read_ptr(read_ptr, ready_in) + i);
            end if;
        end process;
    end generate;

    update_valid_out: process(read_ptr) is
    begin
        if read_ptr + STRIDE_IN_BYTES <= index_t'high then
            valid_out <= '1';
        else
            valid_out <= '0';
        end if;
    end process;


end architecture;
