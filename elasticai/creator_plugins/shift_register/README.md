# Shift Register Plugin

**Translation Stage**: *low level ir* to *vhdl*


This plugin implements a shift register in VHDL. 
The shift register is a digital circuit that can store and shift data.
The plugin is parameterized by the width of the data points (`WIDTH`) and 
the number of data points to make available for reading via the outgoing line `d_out`.

```vhdl
entity shift_register is
    generic (
        DATA_WIDTH: positive; -- size of single data point
        NUM_POINTS: positive  -- number of data points to write in a single step
    );
    port (
        clk : in std_logic;
        d_in : in std_logic_vector(DATA_WIDTH - 1 downto 0);
        d_out : out std_logic_vector(DATA_WIDTH*NUM_POINTS - 1 downto 0);
        valid_in : in std_logic;  -- set to '1' while to write data to the register
        rst : in std_logic;  -- setting to '1' will reset the internal counter and set valid_out to `0`
        valid_out : out std_logic := '0' -- will be '1' when the register is full
    );
end entity;
```
