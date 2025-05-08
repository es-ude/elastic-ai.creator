# Sliding Window Plugin

**Translation Stage**: *low level ir* to *vhdl*


A VHDL implementation of a sliding window mechanism commonly used in convolutional neural networks and signal processing.

## Features

- Configurable input and output widths
- Adjustable stride length
- Reset capability
- Valid input signaling

## Parameters

- `INPUT_WIDTH`: Width of the input data bus
- `OUTPUT_WIDTH`: Width of the output window
- `STRIDE`: Number of positions to move the window (default: 1)

## Usage Example

```vhdl
sliding_window_inst : entity work.sliding_window
    generic map (
        INPUT_WIDTH => 12,
        OUTPUT_WIDTH => 3,
        STRIDE => 1
    )
    port map (
        d_in => input_data,
        d_out => window_output,
        clk => clock,
        valid_in => input_valid,
        rst => reset
    );
```

The sliding window takes a wide input vector and outputs smaller windows by sliding across the input data. For example:

* Input:  `111011101010`
* Windows (stride=1): `111`, `110`, `101`, `011`
* Windows (stride=3): `111`, `011`, `101`, `010`

This is particularly useful for implementing convolution operations in hardware.
