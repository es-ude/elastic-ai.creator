library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;


entity fifo_buffer is
  generic (
    DATA_WIDTH : natural;
    DATA_DEPTH : natural
  );
  port (
    d_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    d_out : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    valid_in : in std_logic;
    valid_out : out std_logic;
    ready_in : in std_logic;
    ready_out : out std_logic;
    clk : in std_logic;
    rst : in std_logic
  );
end entity;

architecture rtl of fifo_buffer is

  type ram_type is array(0 to DATA_DEPTH - 1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal ram: ram_type;
  subtype index_t is natural range ram_type'range;
  signal head : index_t;
  signal tail : index_t;
  signal count : index_t;
  signal count_p1 : index_t;
  signal read_while_write_p1 : std_logic;

  signal ready_out_i : std_logic;
  signal valid_out_i : std_logic;

  function next_index(
    index: index_t;
    ready: std_logic;
    valid: std_logic
  ) return index_t is begin
    if ready = '1' and valid = '1' then
      if index = index_t'high then
        return index_t'low;
      else
        return index + 1;
      end if;
    end if;
    return index;
  end function;


begin

  ready_out <= ready_out_i;
  valid_out <= valid_out_i;

  update_head:
  process (clk, rst) is
  begin
    if rst = '1' then
      head <= index_t'low;
    elsif rising_edge(clk) then
      head <= next_index(head, ready_out_i, valid_in);
    end if;
  end process;

  update_tail:
  process (clk, rst) is
  begin
    if rst = '1' then
      tail <= index_t'low;
    elsif rising_edge(clk) then
      tail <= next_index(tail, ready_in, valid_out_i);
    end if;
  end process;

  proc_ram_write:
  process(clk) is
  begin
    if rising_edge(clk) then
      ram(head) <= d_in;
      d_out <= ram(next_index(tail, ready_in, valid_out_i));
    end if;
  end process;

  proc_count:
  process (head, tail) is
  begin
    if head < tail then
      count <= head - tail + DATA_DEPTH;
    else
      count <= head - tail;
    end if;
  end process;

  proc_count_p1:
  process (clk, rst) is
  begin
    if rst = '1' then
      count_p1 <= 0;
    elsif rising_edge(clk) then
        count_p1 <= count;
    end if;
  end process;

  proc_in_ready:
  process (count) is
  begin
    if count < DATA_DEPTH - 1 then
      ready_out_i <= '1';
    else
      ready_out_i <= '0';
    end if;
  end process;

  proc_read_while_write:
  process (clk, rst) is
  begin
    if rst = '1' then
      read_while_write_p1 <= '0';
    elsif rising_edge(clk) then
        read_while_write_p1 <= '0';
        if ready_out_i = '1' and valid_in = '1'
          and valid_out_i = '1' and ready_in = '1' then
            read_while_write_p1 <= '1';
        end if;
    end if;

  end process;

  proc_update_valid_out:
  process (count, count_p1, read_while_write_p1) is
  begin
   valid_out_i <= '1';

    if count = 0 or count_p1 = 0 then
      valid_out_i <= '0';
    end if;

    if count = 1 and read_while_write_p1 = '1' then
      valid_out_i <= '1';
    end if;

  end process;

end architecture;

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;

entity full_window_read_buffer is
  generic (
    DATA_WIDTH : natural;
    DATA_DEPTH : natural;
    STRIDE : natural;
    OUTPUT_WIDTH : natural
  );
  port (
    d_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    d_out : out std_logic_vector(DATA_WIDTH*OUTPUT_WIDTH- 1 downto 0);
    valid_in : in std_logic;
    valid_out : out std_logic;
    ready_out : out std_logic;
    ready_in : in std_logic;
    clk : in std_logic;
    rst : in std_logic
  );
end entity;


architecture rtl of full_window_read_buffer is
  type data_t is array(0 to DATA_DEPTH-1) of std_logic_vector(DATA_WIDTH - 1 downto 0);
  signal data : data_t;

  subtype index_t is integer range data_t'range;
  signal head : index_t := 0;
  signal tail : index_t := 0;
  signal ready_out_i : STD_LOGIC;
  signal valid_out_i : STD_LOGIC;

begin

  valid_out <= valid_out_i;
  ready_out <= ready_out_i;
  
  update_data_sigs:
  process (clk) is
  begin
    if rising_edge(clk) then
      data(head) <= d_in;
      for i in 0 to OUTPUT_WIDTH - 1 loop
        d_out(DATA_WIDTH*(i+1) - 1 downto i*DATA_WIDTH) <= data(tail+i);
      end loop;
    end if;
  end process;

  update_head:
  process (clk, rst) is
  begin
    if rst = '1' then
      head <= 0;
    elsif rising_edge(clk) then
      if valid_in = '1' and ready_out_i = '1' then
        if 1 + head <= index_t'high then
          head <= head + 1;
        end if;
      end if;
    end if;
  end process;

  update_tail:
  process (clk, rst) is
  begin
    if rst = '1' then
      tail <= 0;
    elsif rising_edge(clk) then
      if valid_out_i = '1' and ready_in = '1' then
        if tail + STRIDE <= index_t'high - OUTPUT_WIDTH then
          tail <= tail + STRIDE;
        end if;
      end if;
    end if;
  end process;

  update_valid_out:
  process (head) is
  begin
    if head = index_t'high then
      valid_out_i <= '1';
    else
      valid_out_i <= '0';
    end if;
  end process;

  update_ready_out:
  process (tail) is
  begin
    if tail = index_t'high - OUTPUT_WIDTH then
      ready_out_i <= '0';
    else
      ready_out_i <= '1';
    end if;
  end process;
  
 
end architecture;



