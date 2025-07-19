library ieee;
USE ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use work.skeleton_pkg.all;

entity addressable_output_buffer is
    generic (
        DATA_WIDTH : positive;
        DATA_IN_DEPTH: positive;
        DATA_DEPTH : positive
    );
    port (
        signal clk : in std_logic;
        signal rst : in std_logic;
        signal valid_in : in std_logic;
        signal valid_out : out std_logic;
        signal address : in std_logic_vector(15 downto 0);
        signal d_in : in std_logic_vector(DATA_WIDTH*DATA_IN_DEPTH - 1 downto 0);
        signal d_out : out std_logic_vector(7 downto 0)
    );
end entity;

architecture rtl of addressable_output_buffer is
    constant REQUIRED_PADDING : natural := 8*size_in_bytes(DATA_WIDTH) - DATA_WIDTH;
    constant padding : std_logic_vector(REQUIRED_PADDING - 1 downto 0) := (others => '0');
    constant MAX_WRITES : natural := DATA_DEPTH/DATA_IN_DEPTH;
    constant DATA_WIDTH_BYTES : natural := size_in_bytes(DATA_WIDTH);
    constant NUM_IN_BYTES : natural := DATA_WIDTH_BYTES*DATA_IN_DEPTH;

    type byte_vector_t is array(natural range <>) of std_logic_vector(7 downto 0);
    type unpadded_in_t is array (0 to DATA_IN_DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
    subtype bytes_data_t is byte_vector_t(0 to DATA_WIDTH_BYTES - 1);
    type input_bytes_matrix_t is array (0 to DATA_IN_DEPTH - 1) of bytes_data_t;
    subtype input_bytes_vector_t is byte_vector_t(0 to DATA_IN_DEPTH*DATA_WIDTH_BYTES - 1);
    subtype ram_t is byte_vector_t(0 to DATA_DEPTH*DATA_WIDTH_BYTES - 1);
    subtype index_t is integer range 0 to ram_t'right;

    signal input_bytes_matrix : input_bytes_matrix_t;
    signal input_bytes_vector : input_bytes_vector_t;
    signal unpadded_in : unpadded_in_t;
    subtype write_count_t is natural range 0 to MAX_WRITES - 1;
    signal write_count : write_count_t := 0;
    signal last_write_count : write_count_t := 0;
    signal ram : ram_t;
    signal intern_valid_out : std_logic;

    signal address_int : index_t;

    function next_write_count(
        count : natural;
        valid_in : std_logic) return natural is
    begin
        if valid_in = '1' and count < MAX_WRITES - 1 then
            return count + 1;
        else
            return count;
        end if;
    end function;

    function get_byte_slice(
        byte_id: integer;
        data: std_logic_vector(DATA_WIDTH - 1 downto 0)
    ) return std_logic_vector is
    begin
        if byte_id < DATA_WIDTH_BYTES - 1 then
            return data(DATA_WIDTH - 1 - byte_id*8 downto DATA_WIDTH - (byte_id + 1)*8);
        else
            return padding & data(DATA_WIDTH - 1 downto 0);
        end if;

    end function;

begin
    address_int <= min_fn(to_integer(unsigned(address)), index_t'high);
    valid_out <= intern_valid_out;

    connect_input_matrix_to_vector:
    for i in 0 to DATA_IN_DEPTH - 1 generate
        input_bytes_vector(i*DATA_WIDTH_BYTES to (i+1)*DATA_WIDTH_BYTES - 1)<=
            input_bytes_matrix(i);
    end generate;

    connect_unpadded_in_to_input_matrix:
    for i in 0 to DATA_IN_DEPTH - 1 generate
        connect_unpadded_in_per_byte:
        for j in 0 to DATA_WIDTH_BYTES - 1 generate
            input_bytes_matrix(i)(j) <= get_byte_slice(j, unpadded_in(i));
        end generate;
    end generate;

    connect_d_in_to_unpadded_in:
    for i in 0 to DATA_IN_DEPTH - 1 generate
        unpadded_in(i) <= d_in((i+1)*DATA_WIDTH - 1 downto i*DATA_WIDTH);
    end generate;

    update_ram:
    process (clk) is
    begin
        if rising_edge(clk) and valid_in = '1' and intern_valid_out = '0' and write_count < MAX_WRITES then
            ram(write_count*NUM_IN_BYTES to (write_count+1)*NUM_IN_BYTES - 1)
                <= input_bytes_vector;
        end if;
    end process;


    update_d_out:
    process (clk) is
    begin
        if rising_edge(clk) then
            d_out <= ram(address_int);
        end if;
    end process;

    update_last_write_count:
    process (clk, rst) is
    begin
        if rst = '1' then
            last_write_count <= 0;
        elsif rising_edge(clk) then
            last_write_count <= write_count;
        end if;
    end process;

    update_write_count:
    process (clk, rst) is
    begin
        if rst = '1' then
            write_count <= 0;
        elsif rising_edge(clk) then
            write_count <= next_write_count(write_count, valid_in);
        end if;
    end process;

    update_valid_out:
    process (write_count, last_write_count) is
    begin
        if write_count =  MAX_WRITES - 1 and last_write_count = MAX_WRITES - 1 then
            intern_valid_out <= '1';
        else
            intern_valid_out <= '0';
        end if;
    end process;
end architecture;
    
