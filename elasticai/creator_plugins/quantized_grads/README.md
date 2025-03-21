# Quantized Grads

This subproject can be used for emulating training in python with quantized gradients.
The quantization is implemented as a fake-quantization. 

## Python
### Layers
Right now this package supports the following base layers:
- [X] Batchnorm2d
- [X] Conv1d
- [X] Conv2d
- [X] Linear
- [X] Sigmoid
- [X] ReLU

Both quantization schemes can be applied for forward and backward independently.
In my opinion the only valid combinations are the following three combinations.

1. Forward Fullresolution / Backward Fullresolution
2. Forward Quantized / Backward Fullresolution
3. Forward Quantized / Backward Quantized

### Quantization
Right now we support following quantization schemes:
- [X] FixedPoint
- [X] FixedPointStochastic
- [ ] BlockFloatingPoint
- [ ] BlockFloatingPointStochastic
