import torch
from torch import Tensor
from torch.nn import MSELoss, Conv1d as TorchConv1d

from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.conv1d import Conv1d
from elasticai.creator_plugins.quantized_grads.linear_quantization.base_modules.linear import Linear
from elasticai.creator_plugins.quantized_grads.linear_quantization.linear_quantization_config import LinearAsymQuantizationConfig, IntQuantizationConfig

from elasticai.creator_plugins.quantized_grads.linear_quantization.module_quantization import \
    ModuleQuantizeLinearAsymForwHTE
from elasticai.creator_plugins.quantized_grads.linear_quantization.param_quantization import QuantizeParamSTEToIntHTE, QuantizeParamSTEToLinearAsymQuantizationHTE


torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    torch.manual_seed(0)
    epochs = 10
    num_bits = 8
    input_quantisation = ModuleQuantizeLinearAsymForwHTE(LinearAsymQuantizationConfig(num_bits=num_bits))
    output_quantisation = ModuleQuantizeLinearAsymForwHTE(LinearAsymQuantizationConfig(num_bits))

    weight_quantization = QuantizeParamSTEToLinearAsymQuantizationHTE(LinearAsymQuantizationConfig(num_bits))
    bias_quantization = QuantizeParamSTEToIntHTE(IntQuantizationConfig(num_bits))
    #l = torch.nn.Sequential(Linear(input_quantization=output_quantisation, output_quantization=output_quantisation, in_features=3, out_features=2, weight_quantization=weight_quantization, bias=True, bias_quantization=bias_quantization))
    #l = torch.nn.Sequential(Linear(input_quantization=output_quantisation, output_quantization=output_quantisation, in_features=3, out_features=4, weight_quantization=weight_quantization, bias=True, bias_quantization=bias_quantization),
    #                           Linear(input_quantization=output_quantisation, output_quantization=output_quantisation, in_features=4, out_features=2, weight_quantization=weight_quantization, bias=True, bias_quantization=bias_quantization))
    torch.manual_seed(0)
    l = torch.nn.Sequential(Conv1d(1,1,2, input_quantization=input_quantisation, output_quantization=output_quantisation, weight_quantization=weight_quantization, bias_quantization= bias_quantization))
    #l = torch.nn.Sequential(TorchConv1d(1, 1, 2))
    optim = torch.optim.SGD(l.parameters(), lr=0.01)
    loss_fn = MSELoss()
    x = Tensor([[3, 2, 1]])
    #x, x_scale, x_zero_point = quantize_linear_asym_hte(x, Tensor([0]), Tensor([255]))
    label = Tensor([[1, 2]])
    l[0].weight.retain_grad()
    print(f"{l[0].weight.grad=}")
    print(f"{l[0].weight=}")
    for i in range(epochs):
        print()
        y = l(x)
        #print(f"{l[0].weight=}")
        #print(f"{l[0].bias=}")
        #print(y)
        loss = loss_fn(y, label)
        print(loss.item())
        loss.backward()
        print(f"{l[0].parametrizations.weight.original=}")
        print(f"{l[0].parametrizations.weight.original.grad=}")
        optim.step()
        optim.zero_grad()


