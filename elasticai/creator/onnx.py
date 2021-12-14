from typing import List

import torch

from elasticai.creator.tags_utils import get_tags


def _preprocess_tags(tags: dict):
    processed_dic = {}
    unprocessed_vals = []
    for key, value in tags.items():
        to_check = value
        if isinstance(value, List):
            to_check = value[0]
        if isinstance(to_check, int):
            processed_dic[key + "_i"] = value
        elif isinstance(to_check, float):
            processed_dic[key + "_f"] = value
        elif isinstance(to_check, str):
            processed_dic[key + "_s"] = value
        else:
            unprocessed_vals.append(value)
    return processed_dic, unprocessed_vals

class AutogradWrapper(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, callable):
        return input

    @staticmethod
    def symbolic(g, x, callable):
        tags = get_tags(callable)
        kwargs,args= _preprocess_tags(tags)
        ret = g.op("custom_ops::Wrapper",*args,operation_name_s = type(callable).__name__,**kwargs )
        return ret
    
    

class ModuleWrapper(torch.nn.Module):
    def __init__(self, module, autograd_fn=AutogradWrapper, ):
        super().__init__()
        self.autograd_fn = autograd_fn
        self.module = module

    def forward(self, input):
        return self.autograd_fn.apply(input, self.module)
