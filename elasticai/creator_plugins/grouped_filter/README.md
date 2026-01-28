# Grouped Filter

**Translation Stage**: *low level ir* to *vhdl*


Use this plugin to generate a grouped filter from multiple filter kernels.
The `grouped_filter` lowering function will consider the filter parameters and the kernels provided in the attribute of an IR node to generate a grouped filter implementation.

## Important note on filter order for `groups > 1`

TLDR: The plugin assumes that data will be presented as a bit vectore, where each channel channel consists
of a single bit, bit order is MSB first and new time steps enter the filter from LSB towards MSB. It follows,
that more significant bit corresponds to lower channel id, if one would map that concept to a machine learning
framework.

There are several dimensions that describe a grouped filter

- `kernel_size`
- `data_width`
- `groups`
- `in_channels`
- `out_channels`

Clients need to ensure the provided filters handle
these dimensions consistently.

The following example shall demonstrate the issue.
We can interpret an eight bit vector as

- `kernel_size=2`
- `data_width=2
- `in_channels=2`

We use the following abbreviations to index these items along these dimensions

- **C**: Channel
- **T**: Time step
- **B**: Bit
- **K**: Kernel

Clients **should not** order data like this

```
| T0                | T1                |
| C0      | C1      | C0      | C1      |
| B1 | B0 | B1 | B0 | B1 | B0 | B1 | B0 |
```

This orders channels in the opposite direction as the data width dimension.
Clients **need to** use the **same direction** for both orders.
The remainder of this text illustrates why.


Aligning channel and bit order means, we do not have to differentiate between data
width and group size. Let's assume a kernel size of 2. Then with

```
| T0                | T1                |
| C1      | C0      | C1      | C0      |
| B1 | B0 | B1 | B0 | B1 | B0 | B1 | B0 |
```

we can model the implementation as either having

data_width = 2, kernel_size=2, in_channels=2, groups=1 (see below)

```
| T0                | T1                |
| G0                | G0                |
| C1      | C0      | C1      | C0      |
| B1 | B0 | B1 | B0 | B1 | B0 | B1 | B0 |
```

or

data_width = 1, kernel_size=2, in_channels=4, groups=2 (see below)

```
| T0                | T1                |
| G1      | G0      | G1      | G0      |
| C3 | C2 | C1 | C0 | C3 | C2 | C1 | C0 |
| B0 | B0 | B0 | B0 | B0 | B0 | B0 | B0 |
```

The second model, allows us to map scenarios with data_width > 1 to implementations
with a single bit data width. Thus, we assume that all data is order most significant
item first and we can freely interpret an n-bit vector as n one-bit channels or
a single n-bit wide channel.