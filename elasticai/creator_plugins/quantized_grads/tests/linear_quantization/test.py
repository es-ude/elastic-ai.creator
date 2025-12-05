import torch
from torch import Tensor
from torch.nn import MSELoss

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear import Linear
from elasticai.creator_plugins.quantized_grads.linear_quantization.linear_quantization_config import LinearQuantizationConfig, IntQuantizationConfig
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_input_quantization import \
    InputQuantizeLinearAsymForwHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_output_quantization import OutputQuantizeLinearAsymForwHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import QuantizeParamSTEToIntHTE, QuantizeParamSTEToLinearQuantizationHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.quantize_linear import quantize_linear_asym_hte, \
    dequantize_linear
torch.autograd.set_detect_anomaly(True)
if __name__ == "__main__":
    torch.manual_seed(0)
    epochs = 220
    num_bits = 8
    input_quantization = InputQuantizeLinearAsymForwHTE(LinearQuantizationConfig(num_bits))
    output_quantisation = OutputQuantizeLinearAsymForwHTE(LinearQuantizationConfig(num_bits))

    weight_quantization = QuantizeParamSTEToLinearQuantizationHTE(LinearQuantizationConfig(num_bits))
    bias_quantization = QuantizeParamSTEToIntHTE(IntQuantizationConfig(num_bits))

    l = Linear(input_quantization=input_quantization, output_quantization=output_quantisation, in_features=3, out_features=2, weight_quantization=weight_quantization, bias=True, bias_quantization=bias_quantization)
    print(l.weight)

    optim = torch.optim.SGD(l.parameters(), lr=0.01)
    loss_fn = MSELoss()
    x = Tensor([3, 2, 1])
    #x, x_scale, x_zero_point = quantize_linear_asym_hte(x, Tensor([0]), Tensor([255]))
    label = Tensor([1, 2])
    for i in range(epochs):
        print()
        y = l(x)
        loss = loss_fn(y, label)
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(loss.item())

