import torch


class AutogradWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, callable):
        return torch.zeros_like(input)

    @staticmethod
    def symbolic(g, x, callable):
        ret = g.op("custom_ops::Wrapper".format(type(callable).__name__), x)
        return ret


class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, autograd_fn=AutogradWrapper):
        super().__init__()
        self.autograd_fn = autograd_fn
        self.module = module

    def forward(self, input):
        return self.autograd_fn.apply(input, self.module)
