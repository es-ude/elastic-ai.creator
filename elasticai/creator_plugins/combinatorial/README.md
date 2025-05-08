# Combinatorial Plugin

**Translation Stage**: *low level ir* to *vhdl*

**Defined `PluginSymbol`s:**

* `clocked_combinatorial`
* `unclocked_combinatorial`

This plugin generates code for two different hardware designs.
It implements the `clocked_combinatorial` and `unclocked_combinatorial` types.
To use these implementations register the functions with
the same name to an `Ir2Vhdl` lowering pass.
The `unclocked_combinatorial` combines other `unclocked_combinatorial` designs.
The `clocked_combinatorial` design supports many other designs.
All designs supported by `clocked_combinatorial` have the same port interface as the `clocked_combinatorial` or `unclocked_combinatorial` design.
They differ only in their generic parameters.

The symbols take an `elasticai.creator.ir2vhdl.Implementation` a corresponding implementation.

```{note}
**Supported node types:**
* `'shift_register'`
* `'clocked_combinatorial'`
* `'sliding_window'`
* `'unclocked_combinatorial'`
* `'striding_shift_register'`
```

```{warning}
The interfaces here are experimental. They are highly likely to change in future versions. E.g., we are currently planning to extend them to include `ready_in`, `ready_out` signals.
Those could be used by a downstream component to control data flow.
```

## Combinatorial

```{note}
**Terminology:**
The terms _upstream_ and _downstream_ denote the direction of data flow for the rest of this document.
The _upstream_ component is the one that sends data to the _downstream_ component.
```

```{list-table}
:header-rows: 1

* - Name
  - Direction
  - Type
  - Description
* - `clk`
  - **in**
  - `std_logic`
  - Clock signal. Will typically connects to the clock signal of the enclosing component.
* - `valid_in`
  - **in**
  - `std_logic`
  - `'1'` on **rising** edge signals valid input data. Processes data only if `valid_in` is `'1'`. Connect this to the `valid_out` of the upstream component.
* - `valid_out`
  - **out**
  - `std_logic`
  - Drive `HIGH`/`'1'` on **rising** edge to signal that your component's output data is valid. Connect this to the `valid_in` of the downstream component.
* - `rst`
  - **in**
  - `std_logic`
  - Asynchronous reset: set to `'1'` to reset the network, set to `'0'` to release the reset and allow processing. Typically connects to the reset signal of the enclosing component.
* - `d_in`
  - **in**
  - `std_logic_vector`
  - Input data window. Connect this to `d_out` of upstream components.
* - `d_out`
  - **out**
  - `std_logic_vector`
  - All output data. Connect this to `d_in` of downstream components.
```

```{warning}
Please note that we will most likely extend the interface above with two more signals, `ready_in` and `ready_out`, in the future.

| Name | Direction | Type | Description |
|------|-----------|------|-------------|
| `ready_out` | **out** | `std_logic` | Drive `HIGH`/`'1'` when your component is ready to accept new input data via `d_in`. Connects to `ready_in` of upstream components. |
| `ready_in` | **in** | `std_logic` | Connects to `ready_out` downstream components. |
```

## Unclocked Combinatorial

The `unclocked_combinatorial` design holds no state.

| Name | Type | Description |
|------|------|-------------|
| `d_in` | `std_logic_vector` | Input data window |
| `d_out` | `std_logic_vector` | All output data |

## Usage Examples

### Basic Setup

First, register the plugin implementations with the IR2VHDL converter:

```python
from elasticai.creator.ir2vhdl import IR2VHDL
from elasticai.creator_plugins.combinatorial import (
    clocked_combinatorial,
    unclocked_combinatorial
)

# Create the IR2VHDL converter
converter = IR2VHDL()

# Register the implementations
converter.register('clocked_combinatorial')(clocked_combinatorial)
converter.register('unclocked_combinatorial')(unclocked_combinatorial)
```

### Clocked Combinatorial Example

Here's how to create a simple network using a clocked combinatorial component:

```python
from elasticai.creator.ir2vhdl import Implementation, Shape, vhdl_node

# Create a network with an 8-bit clocked component
impl = Implementation(name="my_network")
impl.add_node(vhdl_node(
    name="input",
    type="input",
    implementation="",
    input_shape=Shape(8, 1),  # 8-bit input
    output_shape=Shape(8, 1)
))
impl.add_node(vhdl_node(
    name="my_clocked_component",
    type="clocked_combinatorial",
    implementation="clocked_combinatorial",
    input_shape=Shape(8, 1),
    output_shape=Shape(8, 1)
))
impl.add_node(vhdl_node(
    name="output",
    type="output",
    implementation="",
    input_shape=Shape(8, 1),
    output_shape=Shape(8, 1)
))

# Connect the nodes with indices for port mapping,
impl.add_edge(edge("input", "my_clocked_component", tuple()))
impl.add_edge(edge("my_clocked_component", "output", tuple())))

# Convert to VHDL
vhdl_entity_name, vhdl_code = next(converter([impl]))
```

The unclocked versions are created exactly the same way.
Their only difference lies in the generated interfaces and control signals.
