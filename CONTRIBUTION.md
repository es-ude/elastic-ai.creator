# Contribution Guide


## Concepts
The `elasticai.creator` aims to support
    1. the design and training of hardware optimization aware neural networks
    2. the translation of above designs to a neural network accelerator in a hardware definition language
The first point means that the network architecture, algorithms used during forward as well as backward
propagation strongly depend on the targeted hardware implementation.
Since the tool is aimed at researchers we want the translation process to be straight-forward and easy to reason about.
Opposed to other tools (Apache TVM, FINN, etc.) we prefer flexible prototyping and hand-written
hardware definitions over a wide range of supported architectures and platforms or highly scalable solutions.

The code-base is composed out of the following packages
- `file_generation`:
  - write files to paths on hard disk or to virtual paths (e.g., for testing purposes)
  - simple template definition
  - template writer/expander
- `vhdl`:
  - helper functions to generate frequently used vhdl constructs
  - the `Design` interface to facilitate composition of hardware designs
  - basic vhdl design without a machine learning layer counterpart to be used as dependencies in other designs (e.g., rom modules)
  - additional vhdl designs to make the neural network accelerator accessible via the elasticai.runtime
- `base_modules`:
  - basic machine learning modules that are used as dependencies by translatable layers
- `nn`:
  - package for public layer api; hosting translatable layers of different categories
  - layers within a subpackage of `nn`, e.g. `nn.fixed_point` are supposed to be compatible with each other

## Glossary
| name | meaning                                                                                                                                                      |
|------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| x | name for the input parameter of a layer in resemblance of the mathematical notation                                                                          |
| y| name for the result of a layer computed for an input x                                                                                                       |
| total_bits | total number of bits used to represent a datum in vhdl                                                                                                       |
| design | abstraction of a hardware design to support auto-wiring                                                                                                      |
|auto-wiring | the process of programmatically instantiating and connecting hw designs for layers to form a neural network                                                  |
| fxp | abbreviation and prefix for fixed point numbers                                                                                                              |
| base modules | basic neural network layer implementations for pytorch, typically using strategy patterns to allow injection of concrete mathematical operations (`mathops`) |

## Adding a new translatable layer
Adding a new layer involves three main tasks:
1. define the new ml framework module, typically you want to inherit from `pytorch.nn.Module` and optionally use one
        of our layers from `base_module`
   - this specifies the forward and backward pass behavior of your layer
2. define a corresponding `Design` class
   - this specifies
     - the hardware implementation (ie., which files are written to where and what's their content)
     - the interface (`Port`) of the design, so we can automatically combine it with other designs
     - to help with the implementation you can use the template system as well as the `elasticai.creator.vhdl.code_generation` modules
3. define a trainable `DesignCreator`, typically inheriting from the class defined in 1. and implement the `create_design` method which
   a. extracts information from the module defined in 1.
   b. converts that information to native python types
   c. instantiates the corresponding design from 2. providing the necessary data from a.
    - this step might involve calling `create_design` on submodules and inject them into the design from 2.


### Ports and automatically combining layers
The algorithm for combining layers lives in `elasticai.creator.vhdl.auto_wire_protocols`.
Currently, we support two types of interfaces: a) bufferless design, b) buffered design.

b) a design that features its own buffer to store computation results and will fetch its input data from a previous buffer
c) a design without buffer that processes data as a stream, this is assumed to be fast enough such that a buffered design can fetch its input data through a bufferless design

The *autowiring algorithm* will take care of generating vhdl code to correctly connect a graph of buffered and bufferless designs.

A bufferless design features the following signals:

| name |direction | type           | meaning                                         |
|------|----------|----------------|-------------------------------------------------|
| x    | in       |std_logic_vector| input data for this layer                       |
| y    | out      |std_logic_vector| output data of this layer                       |
| clock| in       |std_logic       | clock signal, possibly shared with other layers |


For a buffered design we define the following signals:

| name |direction | type           | meaning                                         |
|------|----------|----------------|-------------------------------------------------|
| x    | in       |std_logic_vector| input data for this layer                       |
| x_address | out | std_logic_vector | used by this layer to address the previous buffer and fetch data, we address per input data point (this typically corresponds to the number of input features) |
| y    | out      |std_logic_vector| output data of this layer                       |
| y_address | in  | std_logic_vector | used by the following buffered layer to address this layers output buffer (connected to the following layers x_address). |
| clock| in       |std_logic       | clock signal, possibly shared with other layers |
|done | out | std_logic | set to "1" when computation is finished |
|enable | in | std_logic | compute while set to "1" |


### Generating Files
#### The Path Module
#### Templates
#### File Hierarchy
The `Design` class provides an abstract object representation of HW designs.
As such a `Design` can generate one or more files each featuring vhdl constructs, e.g., entities.
`Design` instances can be nested.
