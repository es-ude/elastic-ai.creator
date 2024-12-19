
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;


entity padding_adder_per_data_point is
  generic (
    DATA_WIDTH: integer
  );
  port (
    d_in: in std_logic_vector(DATA_WIDTH - 1 downto 0);
    d_out: out std_logic_vector(padded_size(DATA_WIDTH) - 1 downto 0)
  );
end entity;

architecture rtl of padding_adder_per_data_point is
  constant zeros : std_logic_vector(padding_bits(DATA_WIDTH) - 1 downto 0) := (others => '0');

  begin

    connect_less_significant_bytes:
    if size_in_bytes(DATA_WIDTH) > 1 generate
      d_out(d_out'length - 8 - 1 downto 0) <= d_in(d_out'length - 8 - 1 downto 0);
    end generate;

    d_out(d_out'length - 1 downto d_out'length - 8)
      <= zeros & d_in(d_in'length - 1 downto d_out'length - 8);


end architecture;


library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
use work.padding_pkg.all;


entity padder is
  generic (
    DATA_WIDTH : integer; -- <1>
    DATA_DEPTH : integer  -- <2>
  );
  port (
    d_in : in std_logic_vector(DATA_WIDTH * DATA_DEPTH - 1 downto 0);
    d_out : out std_logic_vector(size_in_bytes(DATA_WIDTH) * DATA_DEPTH * 8 - 1 downto 0)
  );
end entity;

architecture rtl of padder is
  begin

    pad_all_data_points:
      for point_id in 0 to DATA_DEPTH - 1 generate begin
        pad_point_i : entity work.padding_adder_per_data_point
          generic map (
            DATA_WIDTH => DATA_WIDTH
            )
          port map (
            d_in => d_in(DATA_WIDTH * (point_id + 1) -1 downto DATA_WIDTH * point_id),
            d_out => d_out(size_in_bytes(DATA_WIDTH) * 8 * (point_id + 1) - 1 downto size_in_bytes(DATA_WIDTH) * 8 * point_id)
            );
      end generate;


end architecture;
