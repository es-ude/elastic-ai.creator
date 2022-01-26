# Elastic.ai Creator

Design, train and compile neural networks optimized specifically for FPGAs.
Obtaining a final model is typically a three stage process.
* design and train it using the layers provided in the `elasticai.creator` package.
* translate the model to a target representation, e.g. VHDL
* compile the intermediate representation with a third party tool, e.g. Xilinx Vivado (TM)

This version currently only supports [Brevitas](https://github.com/Xilinx/brevitas) as a target representation.

The project is part of the elastic ai ecosystem developed by the Embedded Systems Department of the University Duisburg-Essen. For more details checkout the slides at https://www.researchgate.net/publication/356372207_In-Situ_Artificial_Intelligence_for_Self-_Devices_The_Elastic_AI_Ecosystem_Tutorial.


## Table of contents

[[_TOC_]]

## Requirements

The dependencies of the PyTorch implementation of MTLTranslator are listed below:
- torch
- torchvision
- torchaudio
- matplotlib
- numpy
- scikit-learn
- brevitas
- onnx
- onnxoptimizer

### Poetry

#### Using ElasticAICreator as a dependency
Add the following to your `pyproject.toml`
```toml
[[tool.poetry.source]]
name = "elasticai"
url = "https://git@git.uni-due.de/api/v4/projects/5522/packages/pypi/simple"
secondary = true
```
Afterwards you should be able to add the ElasticAICreator to your project like so
```bash
poetry add elasticaicreator
```

#### Developing the ElasticAICreator
For easy creating a virtual environment and having a guaranteed working environment you can use the
[poetry](https://python-poetry.org/) module.

(before installing poetry make sure you have python3.9 installed on your system)

poetry can be installed in the following way:
```bash
pip install poetry
```
if your default python version is not (3.9.*) you need to run the following command first:
```bash
poetry env use python3.9
```

After installing poetry you can create an environment and pull all necessary dependencies by just typing the following
command in the project root folder where the ```poetry.lock``` and ```pyproject.toml``` is located:

```bash
poetry install
```

The installed poetry environment can be activated typing the following command in the project root folder:
```bash
poetry shell
```
You may consider installing poetry plugin in pycharm and adding the created environment

## Installing

If you want to use the elastic.ai Creator Translator in your project you can install it via pip. 
To install from git run the following command: 
```bash
pip install git+https://git.uni-due.de/embedded-systems/artificial-intelligence/toolchain/mtltranslator
```
To build locally in the directory run:
```bash
pip install  .
```

## Structure of the Project

The structure of the project is as follows.
The [qtorch](elasticai/creator/qtorch) folder includes our implementation of quantized PyTorch layer.
The folder [protocols](elasticai/creator/protocols) contains some general protocols for the models and layers which are also used by multiple translated languages.
In the [translator](elasticai/creator/translator) folder there are the modules which can be used for every translation from a pytorch model to a target language.
The subfolder [brevitas](elasticai/creator/brevitas) is for the translation to Brevitas.
Each language we can translate to has folders for unit tests, integration tests and system test. 

## QTorch

QTorch is a library of quantized PyTorch layers. 
The [layers](elasticai/creator/layers.py) file includes all implemented quantized PyTorch layers.
These layers are the followings:
- QConv1d for quantized Conv1d
- QConv2d for quantized Conv2d
- QLinear for quantized Linear
- QLSTMCell for quantized LSTMCell
- QLSTM for stacking QLSTMCell

For the quantization we implemented quantizers:
- Binarize which converts the weights to be -1 or 1
- Ternarize which converts the weights to be -1 or 0 or 1
- QuantizeTwoBit use Residual Quantization to quantize to two bits
- ResidualQuantization converts weights to a bit vector using the residual multilevel binarization method [RebNet, Ghasemzadeh et al. 2018](https://arxiv.org/pdf/1711.01243.pdf)


We wrote tests for the layers which can be found in the [layers_tests](elasticai/creator/qtorch/tests/test_layer.py).
To add constraints on the convolutional and linear layers you can use the [constraints](elasticai/creator/constraints.py) and can easily expand it with more constraints.
We also implemented blocks of convolution and linear layers consisting of the convolution or linear following a batch normalisation and some activation. 
Also consider the [tests](elasticai/creator/qtorch/tests/test_block.py) for the blocks.
For getting more insight in the QTorch library consider the examples in the [examples](elasticai/creator/examples) folder.

## Users guide 

As a user one want to convert an existing pytorch model to one of our languages.
1. Add our translator as a dependency in your project.
2. Instantiate the model. 
3. Optionally you can train it or load some weights. 
4. Input the model in the translation function like shown in the following. 

Please refer to the system test of each language as an example.

## Translating to Brevitas

How to translate a given PyTorch model consisting of QTorch layers to Brevitas?
This is how to translate a given model to a Brevitas model:

```python
from elasticai.creator.brevitas.brevitas_representation import BrevitasRepresentation

converted_model = BrevitasRepresentation.from_pytorch(qtorch_model).translated_model
```
args:
- qtorch_model: a pytorch model (supports most of the [QTorch](https://git.uni-due.de/embedded-systems/artificial-intelligence/toolchain/qtorch) layers and some standard pytorch layers)

returns:
- converted_model: a Brevitas model

Example usages are shown here: [Brevitas system tests](elasticai/creator/systemTests).
We also support to translate a brevitas model to onnx which is shown in the system test.

### Translations

The following QTorch or PyTorch layers are translated to the corresponding Brevitas layers:

- QConv1d to QuantConv1d
- QConv2d to QuantConv2d
- QLinear to QuantLinear
- Binarize to QuantIdentity
- Ternarize to QuantIdentity
- PyTorch MaxPool1d to PyTorch MaxPool1d
- PyTorch BatchNorm1d to PyTorch BatchNorm1d 
- PyTorch Flatten to PyTorch Flatten
- PyTorch Sigmoid to PyTorch Sigmoid

You can find the mappings in [translation_mapping](elasticai/creator/brevitas/translation_mapping.py) and can easily add more PyTorch layers.

### Supported Layers for Brevitas Translation

- QuantConv1d: quantized 1d convolution with weight- and bias-quantization
- QuantConv2d: quantized 2d convolution with weight- and bias-quantization
- QuantLinear: quantized linear layer with weight- and bias-quantization
- QuantIdentity(act_quant=quantizers.BinaryActivation): binary activation layer
- QuantIdentity(act_quant=quantizers.TernaryActivation): ternary activation layer

### Limitations for Brevitas Translation

- we do not support all QTorch layers in the QTorch repository. Not supported layers are:
  - Ternarization with more complex thresholds e.g threshold of 0.1 
  - ResidualQuantization 
  - QuantizeTwoBit 
  - QLSTM
- we do not support all PyTorch layers, but you can easily add them in the [translation_mapping](elasticai/creator/brevitas/translation_mapping.py).

## General Limitations

By now we only support Sequential models for our translations.

## Developers introductory Guide and Glossary

New translation targets should be located in their own folder, e.g. Brevitas for translating from any language to Brevitas.
Workflow for adding a new translation:
1. Obtain a structure, such as a list in a sequential case, which will describe the connection between every component.
2. Identify and label relevant structures, in the base cases it can be simply separate layers.
3. Map each structure to its function which will convert it, like for [example](elasticai/creator/brevitas/translation_mapping.py).
4. Do such conversions.
5. Recreate connections based on 1.

Each sub-step should be separable and it helps for testing if common functions are wrapped around an adapter.

## Model reporter
As part of this repository an utility called [model reporter](elasticai/creator/model_reporter.py) exists which is used to produce a file with the individual identifier from a given dataset. 
It is used for hardware comparisons, where a small subset of the data is used for comparison. 
Example: 
```
ecg-path ground-truth model-prediction
sinus/08219_107.ecg 0 0 
atrial_fibrillation/07879_392.ecg 1 1
sinus/08219_573.ecg 0 0
sinus/04126_477.ecg 0 0
atrial_fibrillation/07162_287.ecg 1 1
atrial_fibrillation/06426_789.ecg 1 1
atrial_fibrillation/05121_664.ecg 1 1
```
This is expected to use a path to help a validator to automatically extract the data from the said path and add the further predictions.
Example usage:
```python
report = ModelReport(model=loaded_model, data=[labels,inputs,data[1]], is_binary=True, threshold=0.5)
report.write_to_csv(path="results.csv")
```

## Tests

Our implementation is fully tested with unit, integration and system tests.
Please refer to the system tests as examples of how to use the Elastic Ai Creator Translator.
You can run one explicit test with the following statement: 

```python -m unittest discover -p "test_*.py" elasticai/creator/translator/path/to/test.py```

If you want to run all unit tests for example, give the path to the unit tests:

```python -m unittest discover -p "test_*.py" elasticai/creator/translator/path/to/language/unitTests/```

You can also run all tests together:

```python -m unittest discover -p "test_*.py" elasticai/creator/translator/path/to/language/```

If you want to add more tests please refer to the [Test Guidelines](test_guidelines.md).

### Brevitas System Tests

The [Brevitas system tests](elasticai/creator/systemTests) can be used as example use cases of our implementation.
We created tests which check the conversion of a model like we would expect our models will look like.
In addition, we also created tests for validating the conversion for trained models or unusual models. 
Note that you have to use your own data set and therefore maybe do some small adaptions by using the training.

### Brevitas Integration Tests

Our  [Brevitas integration tests](elasticai/creator/integrationTests) are focused on testing the conversion of one specific layer. 
We created for all our supported layers a minimal model with this layer included and test its functionality. 

### Brevitas Unit Tests

In addition to system and integration tests we implemented unit tests. 
The unit tests of each module is named like the model but starting with "test_" and can be found in the unitTest folder.
The Brevitas unit tests can be found [here](elasticai/creator/brevitas/unitTests).

## VHDL Generation

We follow the VHDL code specification of IEEE Std 1076-1993.

### Syntax Checking 

[GHDL](https://ghdl.github.io/ghdl/) supports a [syntax checking](https://umarcor.github.io/ghdl/using/InvokingGHDL.html#check-syntax-s) which checks the syntax of a vhdl file without generating code.
The command is as follows:
```
ghdl -s path/to/vhdl/file
```
So, for example for checking the sigmoid source vhdl files in our project we can run:
```
ghdl -s elasticai/creator/vhdl/source/sigmoid.vhd
```
For checking all vhdl files together in our project we can just run:
```
ghdl -s elasticai/creator/**/*.vhd
```

## Developers Guide
* We use [black](https://black.readthedocs.io/en/stable/index.html) for code formatting. For instruction on setup with your IDE please refer to https://black.readthedocs.io/en/stable/integrations/editors.html#editor-integration.
* importing `*` should be avoided in favor of explicit imports

### Test style Guidelines

#### File IO
In general try to avoid interaction with the filesystem. In most cases instead of writing to or reading from a file you can use a StringIO object or a StringReader.
If you absolutely have to create files, be sure to use pythons [tempfile](https://docs.python.org/3.9/library/tempfile.html) module and cleanup after the tests.


#### Diectory structure and file names
Files containing tests for a python module should be located in a test directory for the sake of separation of concerns. 
Each file in the test directory should contain tests for one and only one class/function defined in the module. 
Files containing tests should be named according to the rubric
`test_ClassName.py`.
Next, if needed for more specific tests define a class which is a subclass of unittest.TestCase like [test_brevitas_model_comparison](elasticai/creator/integrationTests/test_brevitas_model_comparison.py) in the integration tests folder. 
Then subclass it, in this class define a setUp method (and possibly tearDown) to create the global environment. 
It avoids introducing the category of bugs associated with copying and pasting code for reuse. 
This class should be named similarly to the file name.
There's a category of bugs that appear if  the initialization parameters defined at the top of the test file are directly used: some tests require the initialization parameters to be changed slightly. 
Its possible to define a parameter and have it change in memory as a result of a test. 
Subsequent tests will therefore throw errors.
Each class contains methods that implement a test. 
These methods are named according to the rubric
`test_name_condition`

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