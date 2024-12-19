
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;


entity padding_remover_per_data_point is
  generic (
    DATA_WIDTH: integer
  );
  port (
    d_in : in std_logic_vector(size_in_bytes(DATA_WIDTH)*8 - 1 downto 0);
    d_out : out std_logic_vector(DATA_WIDTH - 1 downto 0)
  );

end entity;

architecture rtl of padding_remover_per_data_point is
  constant NUM_BYTES : positive := size_in_bytes(DATA_WIDTH);
  constant PADDING_BITS : positive := NUM_BYTES * 8 - DATA_WIDTH;
  constant DATA_BITS_IN_LAST_BYTE : positive := 8 - PADDING_BITS;
  begin

  per_byte:
   for byte_id in NUM_BYTES - 1 downto 0 generate
     last_byte: if byte_id = NUM_BYTES - 1 generate
       d_out(DATA_WIDTH - 1 downto DATA_WIDTH - DATA_BITS_IN_LAST_BYTE)
         <= d_in((byte_id + 1) * 8 - PADDING_BITS - 1 downto byte_id * 8);
     end generate;
     other_bytes: if byte_id < NUM_BYTES - 1 generate
       d_out(DATA_WIDTH - DATA_BITS_IN_LAST_BYTE - byte_id * 8 - 1
             downto DATA_WIDTH - DATA_BITS_IN_LAST_BYTE - (byte_id + 1) * 8)
         <= d_in((byte_id + 1) * 8 - 1 downto byte_id * 8);
     end generate;
   end generate;

end architecture;




library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;



entity padding_remover is
  generic (
    DATA_WIDTH: integer;
    DATA_DEPTH: integer
  );
  port (
    d_in : in std_logic_vector(DATA_DEPTH * size_in_bytes(DATA_WIDTH) * 8 - 1 downto 0);
    d_out : out std_logic_vector(DATA_DEPTH * DATA_WIDTH - 1 downto 0) := (others => '0')
  );
 end entity;


architecture rtl of padding_remover is
begin


  create_padding_remover_per_sample: for data_id in 0 to DATA_DEPTH - 1 generate
    remove_pad_from_data_id: entity work.padding_remover_per_data_point(rtl)
      generic map (DATA_WIDTH => DATA_WIDTH)
      port map (
        d_in =>
          d_in((data_id + 1) * size_in_bytes(DATA_WIDTH) * 8 - 1 downto data_id * size_in_bytes(DATA_WIDTH)* 8),
        d_out => d_out((data_id + 1) * DATA_WIDTH - 1 downto data_id * DATA_WIDTH)
      );
  end generate;

end architecture;
