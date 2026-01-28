# Shift Register Plugin

**Translation Stage**: *low level ir* to *vhdl*

This plugin implements a shift register in VHDL. 
The shift register is a digital circuit that can store and shift data.
The plugin is parameterized by the width of the data points (`DATA_WIDTH`) and 
the number of data points to make available for reading via the outgoing line `d_out`.

A `STRIDE` generic parameter can be set. The shift register will only skip `STRIDE-1` writes, but still signal ready to upstream.
If you are implementing a filter with internal logic you might want to handle stride logic yourself, but for combinatorial components, you can use the following strategy.

To implement a filter with stride $s > 1$ you set the `STRIDE` parameter of the succeeding shift register to $s$.
Thus, the register will store the first filter output, skip $s-1$ outputs and repeat.
While this might seem unintuitive, it is equivalent to the preceeding filter moving by $s$ steps instead of $1$.
