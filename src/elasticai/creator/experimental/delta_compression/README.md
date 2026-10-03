# Delta Compression Plugin

Provides delta compression for neural network weight tensors. Instead of storing each weight as an absolute value, weights are encoded as differences (deltas) from a reference, reducing the effective bit-width needed for transmission or storage.

## Concepts

**Delta encoding** transforms a tensor so that all elements after the first represent differences from a reference. Two strategies are available:

| Class                 | Reference for each element                 |
| --------------------- | ------------------------------------------ |
| `ConsecutiveDelta`    | Previous element (`input[i] - input[i-1]`) |
| `FixedReferenceDelta` | First element (`input[i] - input[0]`)      |

**Compression** clips or masks the resulting deltas to fit a smaller bit-width. Two strategies are available:

| Class                  | Method                                                                                  |
| ---------------------- | --------------------------------------------------------------------------------------- |
| `SaturatedCompression` | Clamps values to `[0 if offset 0 else 2^(offset), 2^(width-1+offset)-1]`, restores sign |
| `BitMaskCompression`   | Applies a bitmask keeping only bits `[offset, offset+width-2]`, restores sign           |

## Building a DeltaCompression

Use `DeltaCompBuilder` to combine a delta strategy with a compression strategy:

```python
from elasticai.creator.experimental.delta_compression import DeltaCompBuilder

dc = (
    DeltaCompBuilder()
    .consecutive_delta()                             # or .fixed_reference_delta()
    .saturated_compression(delta_width=4, offset=0)  # or .clamping_compression(...)
    .build()
)
```

### Parameters

- `delta_width`: number of bits available for each delta value
- `offset`: bit-position shift of the valid range (0 = start from bit 0 (LSB), 1 = start from LSB + 1, etc.)

For `saturated_compression(width=4, offset=0)` the valid absolute range is `[0, 7]`.

## compress / inflate

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0, 4.0])

compressed = dc.compress(x, in_place=False)   # [1.0, 1.0, 1.0, 1.0]  (consecutive deltas)
restored   = dc.inflate(compressed)            # [1.0, 2.0, 3.0, 4.0]
```

`inflate(compress(x)) == x` holds whenever all deltas fit within the compression range. Out-of-range deltas are clipped, making the round-trip lossy.

Both methods accept `in_place: bool` (default `True`). Pass `in_place=False` to leave the input tensor unchanged.

## Emulating delta compression at inference

Two decorator classes wrap `base_modules` layers and replace their weight/bias
quantization step with `quantize → compress → inflate`. This lets you observe the
effect of delta compression on model output without modifying the underlying layer
or its weights.

### Property-based decorators: `@delta_compressed_weights` and `@delta_compressed_bias`

For fine-grained control over which parameters receive delta compression, use the
property-based decorators. These wrap the weight or bias parameter itself, applying
`MathOperations.quantize()` followed by delta compression (compress → inflate) on
each read access. Gradients flow through unchanged via Straight-Through Estimator (STE).

```python
from elasticai.creator.nn.fixed_point.linear.layer import Linear
from elasticai.creator.experimental.delta_compression import (
    DeltaCompBuilder,
    delta_compressed_weights,
    delta_compressed_bias,
)

dc = DeltaCompBuilder().consecutive_delta().saturated_compression(4, 0).build()

# Compress only weights
@delta_compressed_weights(dc)
class MyLinear(Linear):
    pass

# Compress only bias
@delta_compressed_bias(dc)
class MyLinear2(Linear):
    pass

# Compress both with different settings
@delta_compressed_weights(dc_weights)
@delta_compressed_bias(dc_bias)
class MyLinear3(Linear):
    pass

layer = MyLinear(in_features=4, out_features=2, total_bits=8, frac_bits=4)
y = layer(x)   # weight accesses return quantized + delta-compressed values
```

**Key features:**

- Property-based: intercepts parameter reads transparently
- STE-enabled: training-compatible with gradient flow preserved
- Elastic-only: validates that layer has `_operations` attribute
- Granular: apply to weights only, bias only, or both independently

> **Note:** These decorators require elasticai layers with `_operations` attribute.
> Using them on standard PyTorch layers (e.g., `torch.nn.Linear`) raises `TypeError`.

## Architecture

```
DeltaCompression
├── Delta          (abstract)
│   ├── ConsecutiveDelta      – each element minus its predecessor
│   └── FixedReferenceDelta   – each element minus the first element
└── Compression    (abstract)
    ├── SaturatedCompression  – clamp absolute value to [min, max]
    └── BitMaskCompression   – bitwise AND with a bitmask (integer tensors only)

Decorators (inference emulation)
├── @delta_compressed_weights(dc)     – property decorator, wraps weight parameter
└── @delta_compressed_bias(dc)        – property decorator, wraps bias parameter
```
