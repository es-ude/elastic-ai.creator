# Grouped Filter

**Translation Stage**: *low level ir* to *vhdl*


Use this plugin to generate a grouped filter from multiple filter kernels.
The `grouped_filter` lowering function will consider the filter parameters and the kernels provided in the attribute of an IR node to generate a grouped filter implementation.

