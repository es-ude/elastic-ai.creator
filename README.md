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



## Users Guide

### Install

You can install the ElasticAI.creator as a dependency using pip:
```bash
python3 -m pip install "elasticai.creator"
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



