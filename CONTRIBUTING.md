# Contribution Guide


## Table of Contents
- [Contribution Guide](#contribution-guide)
  - [Development Environment](#development-environment)
  - [Pull Requests and Commits](#pull-requests-and-commits)
  - [Documentation](#documentation)
  - [Concepts](#concepts)
  - [Tests](#tests)
  - [Adding a new translatable layer (subject to change)](#adding-a-new-translatable-layer-subject-to-change)
    - [Ports and automatically combining layers (subject to change)](#ports-and-automatically-combining-layers-subject-to-change)

## Development Environment

### uv

We rely on [uv](https://docs.astral.sh/uv/) to manage the venv and dependencies.
Install uv by following their [install guide](https://docs.astral.sh/uv/getting-started/installation/).
Git clone our repository

```bash
$ git clone https://github.com/es-ude/elastic-ai.creator.git
```

or

```bash
$ git clone git@github.com:es-ude/elastic-ai.creator.git
```

move into the just cloned repository and run

```bash
$ uv sync
```

This install all runtime as well as most of the
development dependencies. There are more (optional)
development dependency groups that you can install,
e.g., the `lsp` group containing 
python-language-server and pylsp-mypy

```bash
$ uv sync --group lsp
```

### devenv

[devenv](https://devenv.sh) is a tool for managing and
sharing development environments in a declarative manner.
Advantages are
 * define the development environment once and use it across the whole team
 * a `devenv.lock` file will ensure that all defined program versions are the same for the whole team
 * environment and commands can be easily replicated in the CI pipeline
   * e.g., no need to find out how to install ghdl or other dependencies with github actions, just use the `devenv shell` that you're using locally already

#### Setup

After installing the nix package manager and [nix](https://nix.dev/install-nix)
and [installing devenv](https://devenv.sh/getting-started/#2-install-devenv) you
can startup a development environment with
```bash
$ devenv shell
```
For more convenience we recommend combining this workflow with direnv for
automatic shell activation as explained [here](https://devenv.sh/automatic-shell-activation/).

Devenv will automatically give you access to all other relevant tools

 * uv
 * ghdl for hw simulations (run via rosetta on apple silicon)
 * gtkwave for visualizing waveforms produced by ghdl
 * act for testing github workflows locally
 * and more...

for a full list of installed tools have a look at the `devenv.nix` file.

#### Caveats

IMPORTANT: To use the `elasticai.creator` with cuda you might have to disable `devenv` and call uv from your system environment.
We hope to find a workaround or get a fix from upstream in the near future.


### jujutsu

In case you're using [jj](https://jj-vcs.github.io/jj/latest/) for versioning.
You can use ruff to fix previous changes with by adding the following content to your repo config

```toml
[fix.tools.1-ruff-check]
command = ["ruff", "check", "--fix", "--stdin-filename=$path"]
patterns = [
  "glob:'**/*.py'",
  "glob:'**/*.pyi'",
]

[fix.tools.2-ruff-format]
command = ["ruff", "format", "--stdin-filename=$path"]
patterns = [
  "glob:'**/*.py'",
  "glob:'**/*.pyi'",
]

[fix.tools.alejandra]
command = ["alejandra", "-"]
patterns = [
  "glob:'**/*.nix'",
]

[fix.tools.3-ruff-check]
command = ["uv","run", "ruff", "check", "--select", "I", "--fix", "--stdin-filename=$path"]
patterns = [
  "glob:'**/*.py'",
  "glob:'**/*.pyi'",
]
```

You can access the repository config by running `jj config edit --repo`.
The snippet above will use ruff to check and fix linted problems, then reformat files and then order imports.
This will be applied to each change that is mutable, ie. changes belonging to your own branches.


## Pull Requests and Commits
Use conventional commit types (see [here](https://www.conventionalcommits.org/en/v1.0.0-beta.2/#summary)) especially (`feat`, `fix`) and mark `BREAKING CHANGE`s
in commit messages. The message scope is optional.
Please try to use rebasing and squashing to make sure your commits are atomic.
By atomic we mean, that each commit should make sense on its own.
As an example let's assume you have been working on a new feature `A` and
during that work you were also performing some refactoring and fixed a small
bug that you discovered. Ideally your history would more or less like this:

:::
#### Do
:::

```
* feat: introduce A

  Adds several new modules, that enable a new
  workflow using feature A.

  This was necessary because, ...
  This improves ....


* fix: fix a bug where call to function b() would not return

  We only found that now, because there was no test for this
  bug. This commit also adds a new corresponding test.


* refactor: use an adapter to decouple C and D

  This is necessary to allow easier introduction of
  feature A in a later commit.
```

What we want to avoid is a commit history like that one

:::
#### Don't
:::
```
* feat: realize changes requested by reviewer

* feat: finish feature A

* refactor: adjust adapter

* wip: working on feature A

* fix: fix a bug (this time for real)

* fix: fix a bug

* refactor: had to add an adapter

* fix: fix some typos as requested by reviewer

```

If a commit introduces a new feature, 
it should ideally also contain the test coverage, documentation, etc.
If there are changes that are not directly related to that feature, 
they should go into a different commit.


## Documentation

### Manually

First make sure that the necessary dev dependencies are installed

```bash
$ uv sync
```

Now move to the docs folder and build the documentation

```bash
$ uv run sphinx-build docs build/docs
```

While working on the documentation you can run a server that will automatically rebuild the the docs on every change and serve them under `localhost:8000`.
You can start the server with

```bash
$ uv run sphinx-autobuild docs build/docs
```

If you have question about the markup have a look at the [myst-documentation](https://myst-parser.readthedocs.io/en/latest/index.html).

### With devenv (recommended)

Simply run

```bash
$ devenv up
```


## Concepts
The `elasticai.creator` aims to support
    1. the design and training of hardware optimization aware neural networks
    2. the translation of designs from 1. to a neural network accelerator in a hardware definition language
The first point means that the network architecture, algorithms used during forward as well as backward
propagation strongly depend on the targeted hardware implementation.
Since the tool is aimed at researchers we want the translation process to be straight-forward and easy to reason about.
Opposed to other tools (Apache TVM, FINN, etc.) we prefer flexible prototyping and handwritten
hardware definitions over a wide range of supported architectures and platforms or highly scalable solutions.

The code-base is composed out of the following packages
- `file_generation`:
  - provides a very restricted api to generate files or file subtrees under a given root node and defines a basic template api and data structure, compatible with `file_generation`
  - writes files to paths on hard disk or to virtual paths (e.g., for testing purposes)
  - simple template definition
  - template writer/expander
- `vhdl`:
  - shared code that we use to represent and generate vhdl code
  - helper functions to generate frequently used vhdl constructs
  - the `Design` interface to facilitate composition of hardware designs
  - basic vhdl design without a machine learning layer counterpart to be used as dependencies in other designs (e.g., rom modules)
  - additional vhdl designs to make the neural network accelerator accessible via the elasticai.runtime, also see [skeleton](./elasticai/creator/vhdl/system_integrations/README.md)
- `base_modules`:
  - shared functionality and data structures, that we use to create our neural network software modules
  - basic machine learning modules that are used as dependencies by translatable layers
- `nn`:
  - trainable modules that can be translated to vhdl to build a hardware accelerator
  - package for public layer api; hosting translatable layers of different categories
  - layers within a subpackage of `nn`, e.g. `nn.fixed_point` are supposed to be compatible with each other


### Glossary

 - **fxp/Fxp**: prefix for fixed point
 - **flp/Flp**: prefix for floating point
 - **x**: parameter input tensor for layer with single input tensor
 - **y**: output value/tensor for layer with single output
 - **_bits**: suffix to denote the number of bits, e.g. `total_bits`, `frac_bits`, in python context
 - **_width**: suffix to denote the number of bits used for a data bus in vhdl, e.g. `total_width`, `frac_width`
 - **MathOperations/operations**: definition of how to perform mathematical operations (quantization, addition, matrix multiplication, ...)




## Tests

Our implementation is tested with unit and integration.
You can run one explicit test with the following statement:

```bash
python3 -m pytest ./tests/path/to/specific/test.py
```

If you want to run all tests, give the path to the tests:

```bash
python3 -m pytest ./tests ./elasticai
```

There still are unit tests for specific modules in their respective folders.
Those are subject to be moved.

If you want to add more tests please refer to the Test Guidelines in the following.


### Test Style Guidelines

#### File IO

In general try to avoid interaction with the filesystem. In most cases instead of writing to or reading from a file you can use a StringIO object or a StringReader.
If you absolutely have to create files, be sure to use pythons [tempfile](https://docs.python.org/3.9/library/tempfile.html) module and cleanup after the tests.
In most cases you can use the [`InMemoryPath`](elasticai/creator/file_generation/in_memory_path.py) class to write files to the RAM instead of writing them to the hard disc (especially for testing the generated VHDL files of a certain layer).


#### Directory structure and file names

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


#### Unit tests

In those tests each functionality of each function in the module is tested, it is the entry point  when adding new functions.
It assures that the function behaves correctly independently of others.
Each test has to be fast, so use of heavier libraries is discouraged.
The input used is the minimal one needed to obtain a reproducible output.
Dependencies should be replaced with mocks as needed.

#### Integration Tests

Here the functions' behaviour with other modules is tested.
In this repository each integration function is in the correspondent folder.
Then the integration with a single class of the target, or the minimum amount of classes for a functionality, is tested in each separated file.

#### System tests

Those tests will use every component of the system, comprising multiple classes.
Those tests include expected use cases and unexpected or stress tests.

#### Adding new functionalities and tests required

When adding new functions to an existing module, add unit tests in the correspondent file in the same order of the module, if a new module is created a new file should be created.
When a bug is solved created the respective regression test to ensure that it will not return.
Proceed similarly with integration tests.
Creating a new file if a functionality completely different from the others is created e.g. support for a new layer.
System tests are added if support for a new library is added.

#### Updating tests

If new functionalities are changed or removed the tests are expected to reflect that, generally the ordering is unit tests -> integration tests-> system tests.
Also, unit tests that change the dependencies should be checked, since this system is fairly small the internal dependencies are not always mocked.

references: https://jrsmith3.github.io/python-testing-style-guidelines.html

## Adding a new translatable layer (subject to change)

### General approach

Adding a new layer involves three main tasks:
1. define the new ml framework module, typically you want to inherit from `pytorch.nn.Module` and optionally use one
        of our layers from `base_module`
   - this specifies the forward and backward pass behavior of your layer
2. define a corresponding `Design` class
   - this specifies
     - the hardware implementation (i.e., which files are written to where and what's their content)
     - the interface (`Port`) of the design, so we can automatically combine it with other designs
     - to help with the implementation, you can use the template system as well as the `elasticai.creator.vhdl.code_generation` modules
3. define a trainable `DesignCreatorModule`, typically inheriting from the class defined in 1. and implement the `create_design` method which
   a. extracts information from the module defined in 1.
   b. converts that information to native python types
   c. instantiates the corresponding design from 2. providing the necessary data from a.
    - this step might involve calling `create_design` on submodules and inject them into the design from 2.


### Ports and automatically combining layers (subject to change)
The algorithm for combining layers lives in `elasticai.creator.vhdl.auto_wire_protocols`.
The *autowiring algorithm* will take care of generating vhdl code to correctly connect a graph of buffered and bufferless designs.

### Example

Example steps:
- Create a new folder in `elasticai.creator.nn.<quantization_scheme>`
- Create files: `__init__.py`, `layer.py`, `layer_test.py`, `layer.tpl.vhd`, `design.py`, `design_test.py`


#### VHDL layer template: interface and port description

Currently, we support two types of interfaces: a) bufferless design, b) buffered design.

b) a design that features its own buffer to store computation results and will fetch its input data from a previous buffer
c) a design without buffer that processes data as a stream, this is assumed to be fast enough such that a buffered design can fetch its input data through a bufferless design

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


#### design.py

- needs to inherit from `elasticai.creator.vhdl.design.design.Design`
- typical constructor arguments: num_bits (frac_bits, total_bits), in_feature_num, out_feature_num, weights, bias, name
- `port(self)`: defines the number of inputs and outputs and their data widths
- `save_to(self, destination: Path)`: load a template via `elasticai.creator.file_generation.template.InProjectTemplate` to make text replacements in the VHDL template for filling it with values and parameters; if you have read-only-values (like weights and biases) that you load in the VHDL template, use `elasticai.creator.vhdl.shared_designs.rom.Rom` to create them and call their `save_to()` function

#### layer.py

- Create a layer class which inherits from `elasticai.creator.vhdl.design_creator.DesignCreator` and `torch.nn.Module`
- write `create_design` function that returns a `Design`

#### Base modules

- If you want to define custom mathematic operators for your quantization scheme, you can implement them in `elasticai.creator.nn.<quantization_scheme>._math_operations.py`.
- Create a new file in `elasticai.creator.base_modules` which defines a class inheriting from `torch.nn.Module` and specifies the math operators that you need for your base module
- Your different `layer.py` files for every `elasticai.creator.nn.<quantization_scheme>` can then inherit from `elasticai.creator.base_modules.<module>.py`


