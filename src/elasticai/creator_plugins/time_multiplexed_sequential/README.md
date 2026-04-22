# Time Multiplexed Sequential

**Translation Stage**: *high level ir* to *low level ir*


This plugin creates a time multiplexed sequential model. 
It supports the following IR graph types:

- `sequential`
- `network`

and the following node types

- `grouped_filter`

It will use the kernels and parameters from `grouped_filter` nodes to generate an implementation and apply filters to each time step of the input data on each clock cycle.
It will insert shift registers and sliding input windows as needed.
It supports the following concepts:

- arbitrary strides
- arbitrary number of input and output channels  
- arbitrary 1d kernel sizes with where each data point has a size of 1bit (data width = 1bit)

```{note}
Data widths other than a single bit are not officially supported at the moment, but you might still be able to use them by setting data widths and data depths accordingly.
```

Linear layers are not supported currently, but you can often model them as a convolution of kernel size 1.

## To Do

- [ ] Better Support for data widths other than 1bit
- [ ] Better support for linear layers  
- [ ] Support for skip/residual connections
