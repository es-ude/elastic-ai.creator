# ElasticAI.creator

Design, train and compile neural networks optimized specifically for FPGAs.
Obtaining a final model is typically a three stage process.
* design and train it using the layers provided in the `elasticai.creator.nn` package.
* translate the model to a target representation, e.g. VHDL
* compile the intermediate representation with a third party tool, e.g. Xilinx Vivado (TM)

This version currently only supports parts of VHDL as target representations.

The project is part of the elastic ai ecosystem developed by the Embedded Systems Department of the University Duisburg-Essen. For more details checkout the slides at [researchgate](https://www.researchgate.net/publication/356372207_In-Situ_Artificial_Intelligence_for_Self-_Devices_The_Elastic_AI_Ecosystem_Tutorial).



## Table of contents

- [Users Guide](#users-guide)
  - [Install](#install)
  - [Minimal Example](#minimal-example)
  - [Features](#features)
    - [Supported Network Architectures and Layers](#supported-network-architectures-and-layers)
    - [Planned Network Architectures](#planned-network-architectures-and-layers-supported-in-the-future)
    - [Modules in Development](#modules-in-development)
    - [Deprecated Modules](#deprecated-modules-removal-up-to-discussion)
    - [General Limitations](#general-limitations)
- [Developers Guide](#developers-guide)
  - [Install Dev Dependencies](#install-dev-dependencies)
  - [Conventional Commit Rules](#conventional-commit-rules)
  - [Adding new translation targets](#adding-new-translation-targets)
  - [Syntax Checking](#syntax-checking)
  - [Tests](#tests)
    - [Test Style Guidelines](#test-style-guidelines)
      - [File IO](#file-io)
      - [Directory structure and file names](#directory-structure-and-file-names)
      - [Unit tests](#unit-tests)
      - [Integration tests](#integration-tests)
      - [System tests](#system-tests)
      - [Adding new functionalities and tests required](#adding-new-functionalities-and-tests-required)
      - [Updating tests](#updating-tests)



## Users Guide

### Install

You can install the ElasticAI.creator as a dependency using pip:
```bash
python3 -m pip install elasticai-creator
```

The latest version published on PyPi is the one tagged with:
v0.59.2

Currently, we do not automatically pack and push the code to PyPi.
If you want to make sure to use the latest version from the develop branch, you can install the ElasticAI.creator as a dependency via git:

```bash
python3 -m pip install git+https://github.com/es-ude/elastic-ai.creator.git@develop
```

### Minimal Example

In [examples](examples/minimal_example_FPGA_with_MiddlewareV2.py) you can find a minimal example.
It shows how to use the ElasticAI.creator to define and translate a machine learning model to VHDL. It will save the generated VHDL code to a directory called `build_dir`.
Furthermore, it will generate a skeleton for the Elastic Node V5 that you can use to interface with your machine learning model on the FPGA via a C stub (defined in the [elastic-ai.runtime.enV5](https://github.com/es-ude/elastic-ai.runtime.enV5)).


### Features

#### Supported network architectures and layers

- all sequential network architectures representable with `torch.nn.Sequential`
- fixed-point quantized:
  - layers: linear, linear with batch normalization, LSTM
  - activations: hard sigmoid, hard tanh, ReLU
    - precomputed: sigmoid, tanh, adaptable SiLU


#### Planned network architectures and layers supported in the future

- integer-only linear quantization
- 1D convolutional layers (fixed-point)
- gated recurrent unit (fixed-point)


#### Modules in development:

- `elasticai.creator.nn.fixed_point.conv1d`


#### Deprecated modules (removal up to discussion):

- `elasticai.creator.nn.binary` (binary quantization)
- `elasticai.creator.nn.float` (limited-precision floating-point quantization)
- `elasticai.creator.nn.fixed_point.mac`


#### General limitations

By now we only support sequential models for our translations.
That excludes skip and residual connections.


## Developers Guide

### Install Dev Dependencies

- [poetry](https://python-poetry.org/)
- recommended:
  - [pre-commit](https://pre-commit.com/)
  - [node](https://github.com/nvm-sh/nvm)
  - [commitlint](https://github.com/conventional-changelog/commitlint) to help following our [conventional commit]
  (https://www.conventionalcommits.org/en/v1.0.0-beta.2/#summary) guidelines
poetry can be installed in the following way:
```bash
pip install poetry
poetry install
poetry shell
pre-commit install
npm install --save-dev @commitlint/{config-conventional,cli}
sudo apt install ghdl
```

The repository should work with the following versions:
- Python: >=3.10
- GHDL: >=1.0.0, <=4.1.0
- Node: >=v12.22.9, <=v22.4.1

### Project Structure

All packages and modules fall into one of five main categories and can thus be found in the corresponding package

 - `nn`: trainable modules that can be translated to vhdl to build a hardware accelerator
 - `base_modules`: (non-public) shared functionality and data structures, that we use to create our neural network software modules
 - `vhdl`: (non-public) shared code that we use to represent and generate vhdl code
 - `file_generation`: provide a very restricted api to generate files or file subtrees under a given root node and defines a basic template api and datastructure, compatible with `file_generation`


### Glossary

 - **fxp/Fxp**: prefix for fixed point
 - **flp/Flp**: prefix for floating point
 - **x**: parameter input tensor for layer with single input tensor
 - **y**: output value/tensor for layer with single output
 - **_bits**: suffix to denote the number of bits, e.g. `total_bits`, `frac_bits`, in python context
 - **_width**: suffix to denote the number of bits used for a data bus in vhdl, e.g. `total_width`, `frac_width`
 - **MathOperations/operations**: definition of how to perform mathematical operations (quantization, addition, matrix multiplication, ...)


 ### Adding new modules

 #### Adding new quantization scheme of an existing module
 
 TODO

 #### Adding completely new modules

Example steps:
- Create a new folder in `elasticai.creator.nn.<quantization_scheme>`
- Create files: `__init__.py`, `layer.py`, `layer_test.py`, `layer.tpl.vhd`, `design.py`, `design_test.py`

##### VHDL template

- TODO: ask @glencoe to help describe the interface (x, y, x_address, y_address, buffered, unbuffered)
- TODO: if you want to have a new VHDL template, please ask @SuperChange001

##### design.py

- needs to inherit from `elasticai.creator.vhdl.design.design.Design`
- typical constructor arguments: num_bits (frac_bits, total_bits), in_feature_num, out_feature_num, weights, bias, name
- `port(self)`: defines the number of inputs and outputs and their data widths
- `save_to(self, destination: Path)`: load a template via `elasticai.creator.file_generation.template.InProjectTemplate` to make text replacements in the VHDL template for filling it with values and parameters; if you have read-only-values (like weights and biases) that you load in the VHDL template, use `elasticai.creator.vhdl.shared_designs.rom.Rom` to create them and call their `save_to()` function

##### layer.py

- Create a layer class which inherits from `elasticai.creator.vhdl.design_creator.DesignCreator` and `torch.nn.Module`
- write `create_design` function that returns a `Design`

##### Base modules

- If you want to define custom mathematic operators for your quantization scheme, you can implement them in `elasticai.creator.nn.<quantization_scheme>._math_operations.py`.
- Create a new file in `elasticai.creator.base_modules` which defines a class inheriting from `torch.nn.Module` and specifies the math operators that you need for your base module
- Your different `layer.py` files for every `elasticai.creator.nn.<quantization_scheme>` can then inherit from `elasticai.creator.base_modules.<module>.py`

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
python3 -m pytest ./path/to/specific/test.py
```

If you want to run all tests, give the path to the tests:

```bash
python3 -m pytest ./elasticai
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
