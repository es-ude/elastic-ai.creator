from functools import partial
from typing import Union, Tuple, Sequence

import torch
from torch.nn import Module
from torch import Tensor

from elasticai.creator.tags_utils import has_tag, get_tags, TagWrapper, tag

_precomputable_tag = 'precomputable'

WindowLength1D = int
WindowWidth = int
WindowHeight = int
InputChannels = int
BatchSize = int
InputShape1D = tuple[InputChannels, WindowLength1D]
InputShape2D = tuple[InputChannels, WindowHeight, WindowWidth]
NestedTuple = Union[tuple['float', ...], tuple['NestedTuple', ...]]


class Precomputation:
    def __init__(self, module: Module,
                 input_shape: tuple[int],
                 groups: int,
                 input_domain: NestedTuple) -> None:
        self.module = module
        self.input_shape = input_shape
        self.groups = groups
        self.input_domain = input_domain
        self.input = None
        self.output = None

    @staticmethod
    def from_precomputable_tagged(module: TagWrapper[Module]):
        if not has_tag(module, _precomputable_tag):
            raise TypeError
        info_for_precomputation = get_tags(module)[_precomputable_tag]
        return Precomputation(module=module.unwrap(),
                              **info_for_precomputation)

    def __call__(self) -> tuple[NestedTuple, NestedTuple]:
        """Precompute the input output pairs for the block handed during construction of the object.

        Be aware that this sets the corresponding submodule/block to eval mode. To continue training be sure to set
        it back to training mode again.
        """
        self.module.eval()
        self.input = create_input_tensor(
            input_shape=self.input_shape,
            input_domain=self.input_domain,
            groups=self.groups
        )
        self.output = self.module(input)
        return self.input, self.output


def create_input_tensor(
        input_shape: InputShape1D,
        input_domain: Union[Tensor, NestedTuple],
        groups: int,
):
    raise NotImplementedError


def get_precomputations(module):
    tag_filter = partial(has_tag, tag_name=_precomputable_tag)
    submodules = module.children()
    filtered_submodules = filter(tag_filter, submodules)
    yield from (Precomputation.from_precomputable_tagged(submodule)
                for submodule in filtered_submodules)


def precomputable(module: Union[Module, TagWrapper[Module]],
                  input_shape: Tuple[int],
                  groups: int,
                  input_domain: Sequence[float]) -> TagWrapper[Module]:
    """Add all necessary information to allow later tools to precompute the specified module

    The arguments provided will be used to determine the input data that needs
    to be fed to the module to produce the precomputed input-output table.

            Example:

            model = Sequence(
                precomputable(Sequence(
                    QConv1D(...),
                    BatchNorm1D(...),
                    Binarize(),
                ), input_shape=..., input_domain=..., groups=...),
                precomputable(Sequence(
                    QConv1D(...),
                    BatchNorm1D(...),
                    Binarize(),
                ), input_shape=..., input_domain=..., groups=...),
                MaxPool1D(..),

            precomputations = get_precomputations(model)
            for precompute in precomputations:
                precompute()

            with open('saved_precomputations.txt', 'w') as file:
                for precomputation in precomputations:
                    file.write(precomputation)
    """
    return tag(module, precomputed={
        'input_shape': input_shape,
        'input_domain': input_domain,
        'groups': groups,
    })