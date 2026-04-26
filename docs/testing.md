# Testing

Testing frameworks and methodologies for elasticai.creator.

## HW-in-the-Loop Testing

The HW-in-the-Loop testing framework in elasticai.creator enables testing of hardware implementations by synthesizing VHDL designs and running them on actual hardware devices.

### Key Components

- The `HWTester` class manages hardware testing, including device connections, bitstream uploads, and hardware function execution.
- The `HWTester.prepare_hw_function(src_dir, id=None)` method prepares and uploads a hardware function for testing.

### Workflow

1. Generate VHDL and constraint files
2. Synthesize VHDL into bitstream
3. Connect to hardware device
4. Upload bitstream to FPGA
5. Execute hardware function
6. Retrieve results

### Example

```python
import elasticai.experiment_framework.remote_control as eaixp_rc
import pytest
from elasticai.experiment_framework.synthesis import CachedVivadoSynthesis
from elasticai.creator.testing import HWTester

@pytest.mark.hardware
def test_hardware_function(tmp_path):
    # Setup
    synthesis = CachedVivadoSynthesis()
    device = eaixp_rc.probe_for_devices()[0]
    tester = HWTester(
        synth_fn=synthesis.synthesize,
        device=eaixp_rc.remote_control.connect_remote_control(device),
    )
    
    hw_id = generate_vhdl_files_and_return_id(...)
    
    # Prepare and run hardware function
    with tester.prepare_hw_function(build_dir, id=hw_id) as run_inference:
        predictions = run_inference(b"\x01\x01\x00\x01", 3)
    
    # Verify results
    assert predictions == expected_output_words
```

### Implementation Details

#### Device Management

Uses remote control interface via context manager, without direct dependency on [`elasticai.experiment_framework`](https://github.com/es-ude/elastic-ai.experiment-framework).

#### Hardware Function Identification

- With ID: Checks if function is already loaded (avoids re-upload)
- Without ID (None): Always uploads bitstream

### API Reference

- [Testing Module](../api/testing.md)
- [Core API Reference](../api/core.md)

### Related Projects

- [Elastic-AI Experiment Framework](https://github.com/es-ude/elastic-ai.experiment-framework) - Hardware device control and synthesis tools

### Requirements

- Hardware device connected to system
- Synthesis tools configured (e.g., Vivado)
- Device communication context manager

Use `@pytest.mark.hardware` marker for hardware tests.

### Best Practices

- Use context manager pattern for hardware functions
- Provide hardware function IDs to avoid unnecessary re-uploads

## Simulation

- [Testing Module API](../api/testing.md)
