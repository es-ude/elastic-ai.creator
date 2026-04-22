# Skeleton

**Translation Stage**: *low level ir* to *vhdl*

This skeleton implementation is the middleware-facing top-level adapter for inference. The current architecture is controller-led and frame-oriented:

1. `skeleton` handles the middleware/SPI protocol.
2. `skeleton_inference_controller` orchestrates input buffering, network execution, and output readback.
3. `skeleton_frame_ingress` accepts input bytes and streams logical samples into the controller.
4. `skeleton_network_runner` feeds the network and captures its outputs.

## Generics

When creating a new skeleton instance, set the following generic parameters:

| name | type | meaning |
|---|---|---|
| `DATA_IN_WIDTH` | `int` | Number of bits per input sample |
| `DATA_IN_DEPTH` | `int` | Number of input samples |
| `DATA_OUT_WIDTH` | `int` | Number of bits per output sample |
| `DATA_OUT_DEPTH` | `int` | Number of output samples |

These parameters determine the logical width and depth of the internal buffers and the external byte-addressable transport.

## Padding Behaviour

Input and output transport data are byte-aligned. Logical samples remain `DATA_IN_WIDTH` / `DATA_OUT_WIDTH` wide inside the inference path, while the middleware-facing storage is padded to the next whole byte.

## Network Interface

The network is expected to follow the same port contract used by the `shift_register` plugin:

```vhdl
entity network is
  port (
    CLK : in std_logic;
    D_IN : in std_logic_vector(DATA_WIDTH - 1 downto 0);
    D_OUT : out std_logic_vector(DATA_WIDTH - 1 downto 0);
    SRC_VALID : in std_logic;
    RST : in std_logic;
    VALID : out std_logic;
    READY : out std_logic;
    DST_READY : in std_logic;
    EN : in std_logic
  );
end entity;
```

## Current Scope

The implementation currently supports frame ingress only. Streaming ingress is planned separately.

## Example

For a 1d-cnn processing a time series with three input channels, twelve bits per input channel and time step, and a total of 120 time steps, set `DATA_IN_WIDTH=12` and `DATA_IN_DEPTH=3*120`. The middleware stores transport bytes, while the inference path consumes logical samples at width `DATA_IN_WIDTH` and produces byte-padded output for readback.
