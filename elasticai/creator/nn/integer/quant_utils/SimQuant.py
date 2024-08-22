from torch.autograd import Function


class SimQuant(Function):
    @staticmethod
    def forward(ctx, x_r: float, x_r_QParams) -> float:
        x_q = x_r_QParams.quantize(x_r).to(x_r.device)
        min_float = x_r_QParams.min_float.to(x_r.device)
        max_float = x_r_QParams.max_float.to(x_r.device)

        ctx.save_for_backward(x_r, min_float, max_float)
        return x_r_QParams.dequantize(x_q)

    @staticmethod
    def backward(ctx, grad_output):
        x_r, min_float, max_float = ctx.saved_tensors
        grad_output = grad_output.to(x_r.device)
        grad_input = grad_output.clone()
        grad_input[x_r.gt(max_float)] = 0
        grad_input[x_r.lt(min_float)] = 0
        return grad_input, None, None, None
