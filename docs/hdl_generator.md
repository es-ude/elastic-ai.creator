# HDL Generator Abstraction Layer

A unified interface for generating VHDL and Verilog code, abstracting away the differences between the two hardware description languages.

## Overview

The HDL generator provides a common API for:

- Creating signals/wires
- Instantiating modules/entities
- Managing port connections
- Building parameterized templates

## Usage Examples

### Basic Signal/Wire Creation

```python
from elasticai.creator.hdl_generator import create_generator, HDLLanguage

# Create a generator for VHDL
vhdl_gen = create_generator(HDLLanguage.VHDL)

# Create signals
clk = vhdl_gen.create_signal("clk")  # Single bit
data = vhdl_gen.create_signal("data", width=8)  # 8-bit vector

# Generate definition code
for line in data.define():
    print(line)
# Output: signal data : std_logic_vector(8 - 1 downto 0) := (others => '0');

# Same API works for Verilog
verilog_gen = create_generator(HDLLanguage.VERILOG)
data_verilog = verilog_gen.create_signal("data", width=8)

for line in data_verilog.define():
    print(line)
# Output: wire [7:0] data;
```

### Creating Module/Entity Instances

```python
from elasticai.creator.hdl_generator import create_generator, HDLLanguage
from elasticai.creator.hdl_ir import Shape
from elasticai.creator.ir2vhdl import factory as vhdl_factory
from elasticai.creator.ir import attribute

# Create a generator
gen = create_generator(HDLLanguage.VHDL)

# Create a node representing the module/entity
node = vhdl_factory.node(
    "my_component",
    attribute(),
    type="processor",
    implementation="simple_processor",
    input_shape=Shape(8),
    output_shape=Shape(8),
)

# Create signals
clk = gen.create_null_signal("clk")  # Input port, no definition needed
rst = gen.create_null_signal("rst")
data_in = gen.create_signal("proc_data_in", width=8)
data_out = gen.create_signal("proc_data_out", width=8)

# Create instance
instance = gen.create_instance(
    node=node,
    generics={"DATA_WIDTH": "8", "DEPTH": "16"},
    ports={
        "clk": clk,
        "rst": rst,
        "data_in": data_in,
        "data_out": data_out,
    }
)

# Generate signal definitions
print("-- Signal definitions:")
for line in instance.define_signals():
    print(line)

print("\n-- Component instantiation:")
for line in instance.instantiate():
    print(line)
```

Output:

```vhdl
-- Signal definitions:
signal proc_data_in_my_component : std_logic_vector(8 - 1 downto 0) := (others => '0');
signal proc_data_out_my_component : std_logic_vector(8 - 1 downto 0) := (others => '0');

-- Component instantiation:
my_component: entity work.simple_processor(rtl) 
generic map (
  DATA_WIDTH => 8,
  DEPTH => 16
  )
  port map (
    clk => clk,
    rst => rst,
    data_in => proc_data_in_my_component,
    data_out => proc_data_out_my_component
  );
```

### Template Generation

```python
from elasticai.creator.hdl_generator import create_generator, HDLLanguage

# VHDL Template
vhdl_gen = create_generator(HDLLanguage.VHDL)

vhdl_prototype = '''
entity counter is
    generic (
        WIDTH : natural := 8
    );
    port (
        clk : in std_logic;
        count : out std_logic_vector(WIDTH-1 downto 0)
    );
end entity counter;
'''

template = (
    vhdl_gen.create_template_director()
    .set_prototype(vhdl_prototype)
    .add_parameter("WIDTH")
    .build()
)

# Generate code with specific parameters
code = template.substitute(entity="my_counter", WIDTH="16")
print(code)
```

## Complete Workflow Example

```python
from elasticai.creator.hdl_generator import HDLGenerator, create_generator, HDLLanguage
from elasticai.creator.hdl_ir import Shape
from elasticai.creator.ir import attribute

def generate_hdl_code(generator: HDLGenerator, node_factory) -> str:
    """Generate HDL code using the injected generator and factory."""
    
    # Define the design
    node = node_factory.node(
        "fifo",
        attribute(),
        type="buffer",
        implementation="simple_fifo",
        input_shape=Shape(8),
        output_shape=Shape(8),
    )
    
    # Create signals/wires
    clk = generator.create_null_signal("clk")
    rst = generator.create_null_signal("rst")
    write_data = generator.create_signal("wr_data", width=8)
    read_data = generator.create_signal("rd_data", width=8)
    write_en = generator.create_signal("wr_en")
    read_en = generator.create_signal("rd_en")
    
    # Create instance
    instance = generator.create_instance(
        node=node,
        generics={"DEPTH": "16", "WIDTH": "8"},
        ports={
            "clk": clk,
            "rst": rst,
            "wr_data": write_data,
            "rd_data": read_data,
            "wr_en": write_en,
            "rd_en": read_en,
        }
    )
    
    # Generate code
    lines = []
    lines.extend(instance.define_signals())
    lines.append("")
    lines.extend(instance.instantiate())
    
    return "\n".join(lines)

# Generate VHDL
from elasticai.creator.ir2vhdl import factory as vhdl_factory
vhdl_gen = create_generator(HDLLanguage.VHDL)
vhdl_code = generate_hdl_code(vhdl_gen, vhdl_factory)
print("VHDL:\n", vhdl_code)

# Generate Verilog
from elasticai.creator.ir2verilog import factory as verilog_factory
verilog_gen = create_generator(HDLLanguage.VERILOG)
verilog_code = generate_hdl_code(verilog_gen, verilog_factory)
print("\nVerilog:\n", verilog_code)
```

## API Reference

### HDLGenerator

Protocol defining the abstraction layer interface. Concrete implementations:
- `VHDLGenerator`: VHDL code generation
- `VerilogGenerator`: Verilog code generation

**Methods:**

- `create_signal(name: str, width: int | None = None) -> HDLSignal`: Create signal/wire
- `create_null_signal(name: str) -> HDLSignal`: Create signal that doesn't need definition
- `create_instance(node: Node, generics: dict | None, ports: dict | None) -> HDLInstance`: Create module/entity instance
- `create_template_director() -> HDLTemplateDirector`: Create template builder

### HDLSignal Protocol

**Methods:**

- `name: str`: Signal/wire name (property)
- `define() -> Iterator[str]`: Generate definition code
- `make_instance_specific(instance: str) -> HDLSignal`: Create instance-specific version

### HDLInstance Protocol

**Methods:**

- `name: str`: Instance name (property)
- `implementation: str`: Module/entity name (property)
- `define_signals() -> Iterator[str]`: Generate signal/wire definitions
- `instantiate() -> Iterator[str]`: Generate instantiation code

### HDLTemplateDirector Protocol

**Methods:**

- `set_prototype(prototype: str) -> HDLTemplateDirector`: Set template prototype
- `add_parameter(name: str) -> HDLTemplateDirector`: Add template parameter
- `build() -> Template`: Build final template

