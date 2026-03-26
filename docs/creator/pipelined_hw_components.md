# Pipelined Neural Network HW Components

All components of a neural network that need to be used programmatically, e.g., during code
generation, need to implement the same interface. It consists of the following signals/wires:

- **name**: **bitwidth**
  - meaning
- `CLK`: 1
- `D_IN`: `INPUT_WIDTH`
  - provides input data
- `D_OUT`: `OUTPUT_WIDTH`
- `SRC_VALID`: 1
  - level high if `D_IN` from upstream is valid to be read.
- `VALID`: 1
  - level high if `D_OUT` is valid for downstream.
- `RST`: 1
  - reset all internal state if level high
- `EN`: 1
  - disable all logic if low
- `READY`: 1
  - this component is ready to process valid data from upstream if level high
- `DST_READY`: 1
  - downstream component is ready to process `D_OUT` if level high.


All signals, except for `EN`, are typically triggered synchronously
on rising clock edges.
