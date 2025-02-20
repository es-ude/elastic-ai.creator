# ElasticAI.creator

Design, train and compile neural networks optimized specifically for FPGAs.
Obtaining a final model is typically a three stage process.
* design and train it using the layers provided in the `elasticai.creator.nn` package.
* translate the model to a target representation, e.g. VHDL
* compile the intermediate representation with a third party tool, e.g. Xilinx Vivado (TM)

This version currently only supports parts of VHDL as target representations.

The project is part of the elastic ai ecosystem developed by the Embedded Systems Department of the University Duisburg-Essen. For more details checkout the slides at [researchgate](https://www.researchgate.net/publication/356372207_In-Situ_Artificial_Intelligence_for_Self-_Devices_The_Elastic_AI_Ecosystem_Tutorial).



## Table of contents

- [ElasticAI.creator](#elasticaicreator)
  - [Table of contents](#table-of-contents)
  - [Users Guide](#users-guide)
    - [Install](#install)
    - [Minimal Example](#minimal-example)
    - [Features](#features)
      - [Supported network architectures and layers](#supported-network-architectures-and-layers)
      - [Planned network architectures and layers supported in the future](#planned-network-architectures-and-layers-supported-in-the-future)
      - [Modules in development:](#modules-in-development)
      - [Deprecated modules (removal up to discussion):](#deprecated-modules-removal-up-to-discussion)
      - [General limitations](#general-limitations)
  - [Structure of the Project](#structure-of-the-project)



## Users Guide

### Install

You can install the ElasticAI.creator as a dependency using pip:
```bash
python3 -m pip install "elasticai.creator"
```

On PyPi the latest tagged version is published.

Currently, we do not automatically pack and push the code to PyPi.
If you want to make sure to use the latest version from the main branch, you can install the ElasticAI.creator as a dependency via git:

```bash
python3 -m pip install git+https://github.com/es-ude/elastic-ai.creator.git@main
```

### Minimal Example

In [examples](examples/minimal_example_FPGA_with_MiddlewareV2.py) you can find a minimal example.
It shows how to use the ElasticAI.creator to define and translate a machine learning model to VHDL. It will save the generated VHDL code to a directory called `build_dir`.
Furthermore, it will generate a skeleton for the Elastic Node V5 that you can use to interface with your machine learning model on the FPGA via a C stub (defined in the [elastic-ai.runtime.enV5](https://github.com/es-ude/elastic-ai.runtime.enV5)).


### Features

- Modular architecture for adding new custom VHDL components
- Translation from IR to VHDL ([combinatorial](./docs/creator/plugins/combinatorial.md))
- [Builtin VHDL components](./docs/creator/plugins/vhdl.md):
  - time multiplexed networks
  - counter
  - shift registers
  - sliding window
  - grouped filters

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

## Structure of the Project

The structure of the project is as follows.
The [creator](elasticai/creator) folder includes all main concepts of our project, especially the qtorch implementation which is our implementation of quantized PyTorch layer.
It also includes the supported target representations, like the subfolder [nn](elasticai/creator/nn) is for the translation to vhdl.
Additionally, we have unit and integration tests in the [tests](tests) folder.


