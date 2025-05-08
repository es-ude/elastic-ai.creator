# Striding Shift Register

**Translation Stage**: *low level ir* to *vhdl*


A VHDL implementation of a shift register with configurable stride, useful for implementing strided operations in hardware designs, particularly for neural network architectures.

## Features

- Configurable data width
- Configurable number of points
- Adjustable stride length
- Reset capability (async)
- Valid input/output signaling


This component is particularly useful in neural network hardware implementations where strided operations are common, such as:

- Strided convolutions
- Pooling layers
- Subsampling operations


## Parameters

- `DATA_WIDTH`: Width of each data point
- `NUM_POINTS`: Number of data points to store
- `STRIDE`: Number of clock cycles to wait between shifts (default: 1)

## Operation

The striding shift register extends a basic shift register by adding a stride parameter. When stride > 1, the register only accepts new input every N clock cycles (where N is the stride value). This is particularly useful for implementing strided convolutions or pooling operations in hardware.

## Use case example


The exact intended purpose of the component might not be obvious at first.
Let's assume we want to implement a neural
network where to consecutive filters
$f_1$ and $f_2$ have attributes as follows:

- $f_1$
  - output width: 2 bits (e.g., one per channel)
  - stride: 2
- $f_2$
  - kernel size: 6 bits (e.g., three per channel)

Let's assume further that we have a hardware component for $f_1$ that will process one step of our input signal, per clock cycle.
To provide enough data for $f_2$ we have to provide a buffer with 6 bits.
As $f_1$ has stride 2 we want $f_2$ to see only the results of every second step of $f_1$, i.e., we want to ignore every second clock cycle.

:::{note}
One would think that this leads to a lot of superfluous idle cylces where $f_2$ spends its time waiting for $f_1$.
However, $f_1$ will also have to wait for $f_0$.
Thus, the whole network can at most be as fast as our first layer.
:::

We can visualize the process with the following waveform:

- `DATA_WIDTH => 2`
- `NUM_POINTS => 3`
- `STRIDE => 2`

``` wavedrom

{signal: [
  {name: 'clk', wave: 'p.........'},
  {name: 'f1_valid_out', wave: '01.....0..'},
  {name: 'valid_in', wave: '01.....0..'},
  {name: 'd_in', wave: 'x333333333', data: ['01','11','00','11','10','11']},
  {name: 'f2_valid_in', wave: '0......1..'},
  {name: 'valid_out', wave: '0......1..'},

  {name: 'd_out', wave: 'x.3.3.3.3.', data: ['XXXX01','XX0100', '010010']}
]}
```

- Input data is provided every clock cycle (valid_in = 1)
- Due to stride = 2, the register only shifts on every second clock cycle
- The output becomes valid once the register is filled
- Each output represents 3 consecutive accepted inputs concatenated together, each with a width of 2 bit

