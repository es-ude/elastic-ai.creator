# Hardware Testing Framework Documentation

This documentation describes the hardware testing framework used in elasticai.creator for testing hardware implementations.

## Overview

The hardware testing framework provides a mechanism to test hardware implementations by synthesizing VHDL designs and running them on actual hardware devices. It allows developers to test their hardware functions in a controlled environment that closely mimics real-world deployment conditions.

## Key Components

### 1. HWTesterContext

The `HWTesterContext` class is the main entry point for hardware testing. It manages the connection to hardware devices and handles the process of uploading bitstreams and executing hardware functions.

```python
from elasticai.creator.testing import HWTesterContext
```

#### Constructor Parameters

- `synth_fn`: A function that takes a source directory path and returns a path to the synthesized design
- `device`: A context manager that provides access to the remote control interface for the hardware device

#### Methods

##### `prepare_hw_function(src_dir, id=None)`

This method prepares a hardware function for testing:

- **Parameters**:
  - `src_dir`: Path to the source directory containing the VHDL files
  - `id`: Optional identifier for the hardware function. If None, the framework will always upload the bitstream without checking if it's already loaded.

- **Returns**: A context manager that yields an `AIAccelerator` instance

Example usage:
```python
with ctx.prepare_hw_function(build_dir, id=hw_id) as run_inference:
    result = run_inference(input_data, result_size)
```

### 2. AIAccelerator

The `AIAccelerator` protocol defines the interface for executing inference on hardware:

```python
class AIAccelerator(Protocol):
    def __call__(self, input_data: bytes, result_size: int) -> bytes: ...
```

This protocol ensures that hardware functions can be called consistently regardless of the underlying implementation.

## How It Works

1. **Design Generation**: The user's code generates VHDL and constraint files (typically using a build function like `build_design`) into a specified directory
2. **Synthesis**: The framework uses the provided synthesis function to convert the VHDL source files into a bitstream
3. **Device Connection**: It connects to a hardware device via a remote control interface
4. **Bitstream Upload**: The generated bitstream is uploaded to the FPGA device
5. **Function Execution**: The hardware function is executed with the provided input data
6. **Result Retrieval**: Results are returned from the hardware device

## Example Usage

Here's a typical usage pattern for hardware testing:

```python
import elasticai.experiment_framework.remote_control as eaixp_rc
import pytest
from elasticai.experiment_framework.synthesis import CachedVivadoSynthesis
from elasticai.creator.testing import HWTesterContext

@pytest.mark.hardware
def test_run_minimal_binary_cnn_defined_in_low_level_ir_on_hardware(tmp_path):
    # Setup
    synthesis = CachedVivadoSynthesis()
    device = eaixp_rc.probe_for_devices()[0]
    ctx = HWTesterContext(
        synth_fn=synthesis.synthesize,
        device=eaixp_rc.remote_control.connect_remote_control(device),
    )
    
    hw_id = generate_vhdl_files_and_return_id(...)
    
    # The prepare_hw_function will:
    # 1. Call synthesis.synthesize() to create bitstream from VHDL files
    # 2. Upload the bitstream to hardware
    with ctx.prepare_hw_function(build_dir, id=hw_id) as run_inference:
        predictions = run_inference(b"\x01\x01\x00\x01", 3)
    
    # Verify results
    assert predictions == expected_output_words
```

## Implementation Details

### Device Management

The framework interfaces with hardware devices through the remote control interface, but does not directly depend on the `elasticai.experiment_framework` package. Instead, it expects a context manager providing the necessary remote control functionality.

### Hardware Function Identification

Each hardware function is identified by a unique ID that is passed by the user. If an ID is provided, the framework checks whether that function is already loaded on the device. If the ID is None, the framework will always upload the bitstream without performing this check.


## Testing Configuration

To run hardware tests, you typically need to:
1. Have hardware devices connected to your system
2. Configure appropriate synthesis tools (like Vivado)
3. Provide a context manager for device communication

Tests are marked with `@pytest.mark.hardware` to distinguish them from software-only tests.

## Best Practices

1. Always use the context manager pattern when preparing hardware functions
2. Pass hardware function IDs when possible to avoid unnecessary re-uploads
