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
    Portarser
    PortMaparser
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
    Instance --> Entity
    Instance --> Architecture
    Instance --> Signal
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
```plantuml
@startuml



package hdl {

  class CodeGenerator {
    +{static}int_to_bitstring(number :int, total_bits: int): str
    +{static}float_to_fixed_point_bit_string(number: float, total_bits: int, frac_bits: int): str
    +{static}int_to_hex_string_little_endian(number: int, bits: int): str
  }





  package vhdl {

    package language_structure {
    +class Design<T> {
     +instantiate(): Instance<T>
     +port: Port<T>
     +name: str
    }
    note left: structural representation of\na hw design for the purpose\nof automatic connection and\ninstantiation of hw designs


    +class Port<T> {
      +in_signals: set<T>
      +out_signals: set<T>
      +signals(): set<T>
    }

    +class Instance<T> {
      +name: str
      +design: Design<T>
    }

    Design --* Port
    Design --> Instance
    Instance --o Design
  }

    package designs {
    }

    ~abstract class BaseDesign {
    }
    note bottom: be sure to not let\nport/signal/instance\nleak out of BaseDesign\n or designs package

    ~class VHDLCodeGenerator as "CodeGenerator" {
    }

    +interface Acceptor<T> {
      +accepts(T): bool
    }

    class VHDLSignal as "Signal" {
      +name: str
      +width: int
    }
    VHDLCodeGenerator --> CodeGenerator
    BaseDesign --> Design
    VHDLSignal --|> Acceptor
    designs --|> BaseDesign
    designs --> VHDLSignal
    designs --> VHDLCodeGenerator

  }



  package dataflow {

    +interface Node<T extends Node> {
      +parents(): list<T>
      +children(): list<T>
    }

    +class DataFlowNode<T extends Acceptor> {
        +sinks(): list<Sink<Source<T>>>
        +sources(): list<Source<T>>
        +children(): list<DataFlowNode<T>>
        +parents(): list<DataFlowNode<T>>
        +append(Node<T> child)
        +prepend(Node<T> parent)
        +is_satisfied(): bool
    }


    +class Sink<T extends Acceptor> {
        +source: Optional<Source<T>>
        +data: T
        +owner: DataFlowNode<T>
        +is_satisfied(): bool
        +accepts(Source<T>): bool
    }

    +class Source<T> {
        +data: T
        +sinks: list<Sink<T>>
        +owner: DataFlowNode<T>
    }


    DataFlowNode <-* Sink
    DataFlowNode <-* Source
    Source <--> Sink
    DataFlowNode --|> Node
    Sink --> Acceptor


  }




  package templates {
    abstract class BaseTemplate
  }


}


class mlframework.torch.nn.Linear1D




@enduml
```
## Responsibilities/Packages
- **templates**: streamlined filling of templates
  - Template
  - VHDLTemplate
  - ...
- **parsing**: basic parsing of vhdl files
  - Portarser
  - PortMaparser
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
