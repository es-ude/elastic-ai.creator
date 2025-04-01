# Counter

The plugin provides a counter implemented in vhdl.

:::
#### Interface
:::

```vhdl
entity counter is
    generic (
        MAX_VALUE : natural  -- <1>
    );
    port (
        clk : in std_logic;
        rst : in std_logic;  -- <2>
        enable : in std_logic := '0';  -- <3>
        d_out : out std_logic_vector(clog2(MAX_VALUE+1) - 1 downto 0) := (others => '0') -- <4>
    );
end entity;
```
1. The maximum value of the counter. After reaching this value, the counter will reset to 0 when reaching the end of the process (ie. on the next change of `clk`).
2. Holding the reset signal will reset the counter to 0.
3. The counter will increment on each rising edge of the clock signal while the enable signal is high.
4. The output of the counter as a `std_logic_vector`.
Use `to_unsigned(0, d_out'length)` to turn this into an unsigned number again.

