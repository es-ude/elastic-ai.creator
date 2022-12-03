import torch
from torch.fx import Tracer as fxTracer


class Tracer(fxTracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return m.__module__.startswith("elasticai.creator.vhdl.modules")
