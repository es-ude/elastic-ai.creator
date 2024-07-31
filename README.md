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
  - [General Limitations](#general-limitations)
- [Structure of the Project](#structure-of-the-project)
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


### Minimal Example

The following example shows how to use the ElasticAI.creator to define and translate a machine learning model to VHDL. It will save the generated VHDL code to a directory called `build_dir`.

```python
from elasticai.creator.nn import Sequential
from elasticai.creator.nn.fixed_point import Linear, HardSigmoid
from elasticai.creator.file_generation.on_disk_path import OnDiskPath


def main() -> None:
    # Define a model
    model = Sequential(
        Linear(in_features=10, out_features=2, bias=True, total_bits=16, frac_bits=8),
        HardSigmoid(total_bits=16, frac_bits=8),
    )

    # Train the model
    run_training(model)

    # Save the VHDL code of the trained model
    destination = OnDiskPath("build_dir")
    design = model.create_design("my_model")
    design.save_to(destination)


if __name__ == "__main__":
    main()
```


### General Limitations

By now we only support Sequential models for our translations.



## Structure of the Project

The structure of the project is as follows.
The [creator](elasticai/creator) folder includes all main concepts of our project, especially the qtorch implementation which is our implementation of quantized PyTorch layer.
It also includes the supported target representations, like the subfolder [nn](elasticai/creator/nn) is for the translation to vhdl.
Additionally, we have unit and integration tests in the [tests](tests) folder.



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
