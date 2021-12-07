from functools import partial
from typing import Tuple, overload, Type

from tags_utils import has_tag, get_tags, TagWrapper
from torch.nn import Module

_precomputable_tag = 'precomputable'


class Precomputation:
    def __init__(self, module: Module,
                 input_shape: Tuple[int],
                 groups: int,
                 input_domain: Tuple[float]) -> None:
        self.module = module
        self.input_shape = input_shape
        self.groups = groups
        self.input_domain = input_domain

    @staticmethod
    def from_precomputable_tagged(module: TagWrapper[Module]):
        if not has_tag(module, _precomputable_tag):
            raise TypeError
        info_for_precomputation = get_tags(module)[_precomputable_tag]
        return Precomputation(module=module.unwrap(),
                              **info_for_precomputation)


def precomputations(module):
    tag_filter = partial(has_tag, tag_name=_precomputable_tag)
    submodules = module.children()
    filtered_submodules = filter(tag_filter, submodules)
    yield from (Precomputation.from_precomputable_tagged(submodule)
                for submodule in filtered_submodules)
