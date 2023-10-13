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
|flp | abbrev. and prefix for floating point |
|`_width` | suffix to denote the number of bits used for a signal in vhdl |
| base modules | basic neural network layer implementations for pytorch, typically using strategy patterns to allow injection of concrete mathematical operations (`mathops`) (quantization, addition, matrix multiplication, ...)|



### Install Dev Dependencies

- [poetry](https://python-poetry.org/)
- recommended:
  - [pre-commit](https://pre-commit.com/)
  - [commitlint](https://github.com/conventional-changelog/commitlint) to help following our [conventional commit](https://www.conventionalcommits.org/en/v1.0.0-beta.2/#summary) guidelines
poetry can be installed in the following way:
```bash
pip install poetry
poetry install
poetry shell
pre-commit install
npm install --save-dev @commitlint/{config-conventional,cli}

# Optional:
sudo apt install ghdl
```


### Conventional Commit Rules

We use conventional commits (see [here](https://www.conventionalcommits.org/en/v1.0.0-beta.2/#summary)). The following commit types are allowed. The message scope is optional.

| Commit Types |
|--------------|
| feat         |
| fix          |
| docs         |
| style        |
| refactor     |
| revert       |
| chore        |
| wip          |
| perf         |



### Tests

Our implementation is tested with unit and integration.
You can run one explicit test with the following statement:

```bash
python3 -m pytest ./tests/path/to/specific/test.py
```

If you want to run all tests, give the path to the tests:

```bash
python3 -m pytest ./tests
```

If you want to add more tests please refer to the Test Guidelines in the following.


#### Test Style Guidelines

##### File IO

In general try to avoid interaction with the filesystem. In most cases instead of writing to or reading from a file you can use a StringIO object or a StringReader.
If you absolutely have to create files, be sure to use pythons [tempfile](https://docs.python.org/3.9/library/tempfile.html) module and cleanup after the tests.
In most cases you can use the [`InMemoryPath`](elasticai/creator/file_generation/in_memory_path.py) class to write files to the RAM instead of writing them to the hard disc (especially for testing the generated VHDL files of a certain layer).


##### Directory structure and file names

Files containing tests for a python module should be located in a test directory for the sake of separation of concerns.
Each file in the test directory should contain tests for one and only one class/function defined in the module.
Files containing tests should be named according to the rubric
`test_<class_name>.py`.
Next, if needed for more specific tests define a class. Then subclass it.
It avoids introducing the category of bugs associated with copying and pasting code for reuse.
This class should be named similarly to the file name.
There's a category of bugs that appear if  the initialization parameters defined at the top of the test file are directly used: some tests require the initialization parameters to be changed slightly.
Its possible to define a parameter and have it change in memory as a result of a test.
Subsequent tests will therefore throw errors.
Each class contains methods that implement a test.
These methods are named according to the rubric
`test_<name>_<condition>`


##### Unit tests

In those tests each functionality of each function in the module is tested, it is the entry point  when adding new functions.
It assures that the function behaves correctly independently of others.
Each test has to be fast, so use of heavier libraries is discouraged.
The input used is the minimal one needed to obtain a reproducible output.
Dependencies should be replaced with mocks as needed.

##### Integration Tests

Here the functions' behaviour with other modules is tested.
In this repository each integration function is in the correspondent folder.
Then the integration with a single class of the target, or the minimum amount of classes for a functionality, is tested in each separated file.

##### System tests

Those tests will use every component of the system, comprising multiple classes.
Those tests include expected use cases and unexpected or stress tests.

##### Adding new functionalities and tests required

When adding new functions to an existing module, add unit tests in the correspondent file in the same order of the module, if a new module is created a new file should be created.
When a bug is solved created the respective regression test to ensure that it will not return.
Proceed similarly with integration tests.
Creating a new file if a functionality completely different from the others is created e.g. support for a new layer.
System tests are added if support for a new library is added.

##### Updating tests

If new functionalities are changed or removed the tests are expected to reflect that, generally the ordering is unit tests -> integration tests-> system tests.
Also, unit tests that change the dependencies should be checked, since this system is fairly small the internal dependencies are not always mocked.

references: https://jrsmith3.github.io/python-testing-style-guidelines.html

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

The `Design` class provides an abstract object representation of HW designs.
As such a `Design` can generate one or more files each featuring vhdl constructs, e.g., entities.
`Design` instances can be nested.

**Important**:
 - All generated design units live in the same library and are referenced via the `work` library name.
 - Our current auto-wiring algorithms, assume that each `Design` object is used by at most one other `Design` object.
 - Never call a `Design`'s constructor from another `Design`. Instead, create a new `Design` object at the caller site,
   e.g., in the `create_design` method of the translatable layer and pass that object to the other design, that should
   use it.
 - If you need to generate vhdl design units, that do not need to be compatible with the auto-wiring algorithms, just
   generate/save the corresponding files in the parent `Design`'s `save_to` method. Be aware that you're responsible for
   avoiding naming conflicts. You can achieve that by combining that units name with the `Design`'s name, that was generated
   by the auto-wiring algorithm, thus ensuring uniqueness.
