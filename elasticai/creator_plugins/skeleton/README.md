# Skeleton

**Translation Stage**: *low level ir* to *vhdl*

This skeleton implementation and the buffered network wrapper act as an adapter between the middleware and the new axi-like interfaces for layers.
It will read the input data from the middleware, remove padding, store it, feed it to the network, and write the padded output data back to the middleware.
Additionally, it takes care of enabling and resetting the network.

When creating a new skeleton instance, you need to set the following generic parameters:


| name              | type   | meaning                             |
|-------------------|--------|-------------------------------------|
| `DATA_IN_WIDTH`   | `int`  | Number of bits per input sample     |
| `DATA_IN_DEPTH`   | `int`  | Number of input samples             |
| `DATA_OUT_WIDTH`  | `int`  | Number of bits per output sample    |
| `DATA_OUT_DEPTH`  | `int`  | Number of output sample             |

These parameters are needed to assign correct sizes to data buffers and data lines.

## Padding Behaviour

The exact padding behaviour depends on the `DATA_*` parameters.
We will pad each data point of size `DATA_IN_WIDTH` from the left with zeros to reach multiples of 8 bit.
Padding will be added to the output data vice versa.

The buffer will store unpadded data.



```{important}
The current implementation assumes that all data is provided at once instead of as a stream. I.e., all network buffers will be invalidated and inference is stopped after processing `DATA_IN_DEPTH` number of `DATA_IN_WIDTH` size samples.
Additionally, we assume all data will be stored in a single buffer of `DATA_IN_DEPTH * DATA_IN_WIDTH` number of bits.

To support a different inference strategy, where we can provide samples indefinitely
```


Most of the logic that does not directly deal with the middleware lives in the *Buffered Network Wrapper*.


## Buffered network wrapper

This component handles preparing the data for the network and providing access to the network output in the correct format.

It will:

1. receive the input data from the middleware
2. pipe that data into a buffer, removing padding in the process
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

```vhdl
signal ai_input_window : std_logic_vector(KERNEL_SIZE - 1 downto 0);
signal d_out_network : std_logic_vector(DATA_OUT_WIDTH - 1 downto 0);

```

```vhdl
network_i: entity work.network(rtl)
    port map (
        clk => clk,
        valid_in => valid_in_network,
        d_in => ai_input_window,
        d_out => d_out_network,
        rst => rst,
        valid_out => valid_out_network
    );
```

Make sure to define your network as such.
:::

### Network interface

The wrapper expects the network to define the following interface:

```{list-table}
:header-rows: 1

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
```

## Known Issues

* The final shift register is always created, even in cases
  where the network does not need it, because the result
  can be read from the last layer directly.
  In the future we could just check whether network output
  dimension is the same as result dimension and skip
  the shift register in that case.

# Example

Assuming we have a 1d-cnn processing a time series with three input channels, twelve bit per input channel and time step and a total of 120 time steps,  we would set `DATA_IN_WIDTH=12` and `DATA_IN_DEPTH=3*120`.
When receiving that data from the *middleware* we expect it to be padded to result in a sample of 360 points, each having a size of 16 bit.
Before that data is provided to the network it will be stored in a buffer of
$3\cdot120\cdot12$ bit. The `KERNEL_SIZE` and `STRIDE` parameters of the *buffered network wrapper* together with the `DATA_IN_*` determine how this input will be fed to the network component.
The network is connected to the above buffer using a sliding window of `KERNEL_SIZE` performing steps of size `STRIDE` per clock cycle until the end of the buffer is reached.
