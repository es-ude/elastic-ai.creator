# Padding plugin

**Translation Stage**: *low level ir* to *vhdl*

The plugin adds two vhdl hardware designs.
Both rely completely on combinational logic.

## Padder

Use the `padder` to pad data in an `std_logic_vector` to bytes.

```vhdl
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
```
1. The number of bits you want to use for each data point in the input data. 
2. The number of data points that make up the input data


:::{admonition} Example
* The sequence of bits `1010` is padded to `00001010` when `DATA_WIDTH` is 4 and `DATA_DEPTH` is 1.
* The sequence of bits `1010 is padded to `00000010_000010` when `DATA_WIDTH` is 2 and `DATA_DEPTH` is 2.
:::

## Padding Remover

The `padding_remover` applies the reverse operation of the `padder`.
It will strip padding from the input data.

```vhdl

entity padding_remover_per_data_point is
  generic (
    DATA_WIDTH: integer
  );
  port (
    d_in : in std_logic_vector(size_in_bytes(DATA_WIDTH)*8 - 1 downto 0);
    d_out : out std_logic_vector(DATA_WIDTH - 1 downto 0)
  );

end entity;
```