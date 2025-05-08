# Lutron Plugin

**Translation Stage**: *low level ir* to *vhdl*


The Lutron plugin provides functionality to generate VHDL code for lookup tables (LUTs). It allows you to define truth tables that map input values to output values.

## Overview

Lutron generates a VHDL entity that implements a lookup table with:

- Configurable input and output widths
- User-defined truth table mappings
- Complete VHDL implementation using a case statement

The truth table entries are tuples of strings containing binary values. Each string must match the specified input_size/output_size.

## Example

Here's a complete example implementing a 2-bit input to 2-bit output LUT:

```python
from elasticai.creator.ir2vhdl import Implementation
from elasticai.creator_plugins.lutron.lutron import lutron

# Define a 2-to-2 bit LUT
impl = Implementation(
    name="lut_0",
    type="lutron",
    data={
        "input_size": 2,
        "output_size": 2,
        "truth_table": (
            ("00", "11"),  # When input is 00, output is 11
            ("01", "10"),  # When input is 01, output is 10
            ("10", "01"),  # When input is 10, output is 01
            ("11", "11"),  # When input is 11, output is 11
        ),
    },
)

# Generate VHDL code
name, vhdl_lines = lutron(impl)
```

The generated VHDL code will create an entity with:

- A std_logic_vector input port `d_in` of width `input_size`
- A std_logic_vector output port `d_out` of width `output_size`
- A case statement that implements the truth table logic
- Proper handling of undefined input cases

## Loading as a Plugin

The Lutron plugin can be loaded dynamically using elasticai's plugin system. This allows you to use Lutron without directly importing it:

```python
from elasticai.creator.ir2vhdl import Ir2Vhdl
from elasticai.creator.ir2vhdl import Implementation
from elasticai.creator.plugin import PluginLoader

# Initialize the lowering pass and plugin loader
lowering = Ir2Vhdl()
plugin_loader = PluginLoader(lowering)

# Load the Lutron plugin
plugin_loader.load_from_package("lutron")

# Create implementation as before
impl = Implementation(
    name="my_lut",
    type="lutron",
    data={
        "input_size": 2,
        "output_size": 2,
        "truth_table": (
            ("00", "11"),
            ("01", "10"),
            ("10", "01"),
            ("11", "11"),
        ),
    },
)

# Generate VHDL using the lowering pass with loaded plugin
name, vhdl_lines = next(lowering([impl]))
```

When using the plugin system, Lutron is automatically registered with the lowering pass and can be used by specifying `type="lutron"` in your Implementation object.


