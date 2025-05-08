# Skeleton

**Translation Stage**: *low level ir* to *vhdl*


This skeleton implementation acts as an adapter between the middleware and the new axi-like interfaces for layers.
It will read the input data from the middleware, store it, feed it to the network, and write the output data back to the middleware.
Additionally, it takes care of enabling and resetting the network.
Most of the logic that does not directly deal with the middleware lives in the *Buffered Network Wrapper*.

## Buffered network wrapper

This component handles preparing the data for the network and providing access to the network output in the correct format.

It will:

1. read the buffered input data from the middleware
2. remove padding from the input data
3. use a sliding window to feed the buffered data to the network
4. pad output data for the middleware

:::{important} Connecting to a neural network

The wrapper instantiates the network as

```vhdl

entity buffered_network_wrapper is
  generic (
    DATA_IN_WIDTH : integer;
    DATA_IN_DEPTH : integer;
    DATA_OUT_WIDTH : integer;
    DATA_OUT_DEPTH : integer;
    KERNEL_SIZE : integer;
    STRIDE : integer
    );

  port (
    signal clk : in std_logic;
    signal valid_in : in std_logic;
    signal rst : in std_logic;
    signal d_in : in std_logic_vector(DATA_IN_DEPTH * size_in_bytes(DATA_IN_WIDTH) * 8 - 1 downto 0);
    signal d_out : out std_logic_vector(DATA_OUT_DEPTH * size_in_bytes(DATA_OUT_WIDTH) * 8 - 1 downto 0);
    signal valid_out : out std_logic
    );
end entity;

```

Make sure to define your network as such.
:::

### Network interface

The wrapper expects the network to define the following interface:

:::{list-table}
* - Name
  - Direction
  - Type
  - Description
* - `clk`
  - `in`
  - `std_logic`
  - Clock Signal
* - `valid_in`
  - `in`
  - `std_logic`
  - `'1'` on *rising* edge signals valid input data

* - `valid_out`
  - `out`
  - `std_logic`
  - `'1'` on *rising* edge signals valid output data

* - `rst`
  - `in`
  - `std_logic`
  - asynchronous reset
    * set to `'1'` to reset the network
    * set to `'0'` to release the reset and allow processing

* - `d_in`
  - `in`
  - `std_logic_vector`
  - input data window

* - `d_out`
  - `out`
  - `std_logic_vector`
  - all output data
:::

## Known Issues

* The final shift register is always created, even in cases
  where the network does not need it, because the result
  can be read from the last layer directly.
  In the future we could just check whether network output
  dimension is the same as result dimension and skip
  the shift register in that case.