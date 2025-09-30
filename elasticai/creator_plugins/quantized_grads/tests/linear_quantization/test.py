import torch
from torch import Tensor
from torch.nn import MSELoss

from elasticai.creator_plugins.quantized_grads.linear_quantization import QuantizeForwHTEBackwHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear import Linear
from elasticai.creator_plugins.quantized_grads.linear_quantization.autograd import QuantizeForwHTEAutograd
from elasticai.creator_plugins.quantized_grads.linear_quantization.linear_quantization_config import \
    LinearQuantizationConfig, IntQuantizationConfig
from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import QuantizeForwHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import \
    QuantizeParamSTEToLinearQuantizationHTE, QuantizeParamToIntHTE, QuantizeParamToLinearQuantizationHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.quantize_linear import quantize_linear_hte_fake

if __name__ == "__main__":
    torch.manual_seed(0)
    epochs = 220
    num_bits = 8
    math_ops = QuantizeForwHTE(LinearQuantizationConfig(num_bits))
    math_ops = QuantizeForwHTEBackwHTE(LinearQuantizationConfig(num_bits), LinearQuantizationConfig(4))

    weight_quantization = QuantizeParamToLinearQuantizationHTE(LinearQuantizationConfig(num_bits))
    bias_quantization = QuantizeParamToIntHTE(IntQuantizationConfig(num_bits))

    l = Linear(math_ops, in_features=3, out_features=2, weight_quantization=weight_quantization, bias=True, bias_quantization=bias_quantization)
    print()

    optim = torch.optim.SGD(l.parameters(), lr=0.01)
    loss_fn = MSELoss()
    x = Tensor([3, 2, 1])
    x = quantize_linear_hte_fake(x, Tensor([0]), Tensor([255]))
    label = Tensor([1, 2])
    for i in range(epochs):
        print()
        y = l(x)

        loss = loss_fn(y, label)
        loss.backward()
        print(f"LinearQ epoch {i}, loss: {loss.item()}")
        print(f"{l.weight.grad=}")
        print(f"{l.bias.grad=}")
        optim.step()
        optim.zero_grad()




