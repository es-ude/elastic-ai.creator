# Architecture
## elasticai.creator.vhdl
### Dependencies
```mermaid
flowchart TB
  transmodules --> modules
  parsing --> language
  designs --> parsing
  designs --> templates
  designs --> language
  designs --> dataflow
  designs --> vhdl_code_generation
  vhdl_code_generation --> code_generation
  transmodules --> designs
  dataflow --> sink
```

### Dependencies with example classes
```mermaid
flowchart TB
  subgraph templates
    VHDLTemplate --> Template
  end
  subgraph designs
    designFPHardSigmoidWithLookUp["FPHardSigmoidWithLookUp"]
  end
  subgraph parsing
    PortParser
    PortMapParser
  end
  subgraph modules
    LSTM
    Linear
  end
  subgraph transmodules["translatable modules"]
    FPHardSigmoidWithLookUp
  end
  transmodules --> modules
  subgraph language
    Architecture --> Entity --> Port --> Signal
    Architecture --> PortMap --> Signal
  end
  parsing --> language
  designs --> parsing
  designs --> templates
  designs --> language
  designs --> dataflow
  transmodules --> designs
  subgraph codegen["code generation"]
    int_to_padded_hex
    int_to_binary
  end
  subgraph vhdlcodegen["vhdl code generation"]
    generate_port_map
    generate_port
    define_signal
  end
  subgraph dataflow
    SinkNode <--> DataFlowNode <--> SourceNode <--> SinkNode
  end

  vhdlcodegen --> codegen
  templates --> vhdlcodegen
  SinkNode --> acceptor["AcceptorInterface"]
  Signal -.-> acceptor
```

## Responsibilities/Packages
- **templates**: streamlined filling of templates
  - Template
  - VHDLTemplate
  - ...
- **parsing**: basic parsing of vhdl files
  - PortParser
  - PortMapParser
  - ...
- **modules**: Trainable Machine Learning Module
  - LSTM
  - LinearLayer
  - ...
- **translatable modules**: combines designs and modules to create modules that can be translated to VHDL
  - FPHardSigmoidWithLookupTable
  - ...
- **language**: very basic structural representation of vhdl language concepts,
  exposing information needed by other components for auto-wiring vhdl
  - Entity
  - Port
  - Signal
  - ...
- **designs**: hardware designs, some of which use graph structures for auto-wiring components
  - MonotonousLookupTableBasedFPFunction
  - Precomputed1DConv
  - XNORPopCount1DConv
  - AcceleratorContainer
- **code generation**: basic code generation functions, e.g., to convert basic data types to strings
  - int_to_binary_string
  - create_block, block with configurable begin and end, that encloses subsequent code
  - int_to_hex_string
  - padded_hex_string_from_int
  - ...
- **vhdl code generation**: basic code generation functions that are specific to vhdl code
  - port_map_from_signals
  - port_from_signals
  - signal_definition
  - signal_connection
