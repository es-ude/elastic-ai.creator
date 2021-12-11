from functools import partial
from typing import Union, Tuple, Sequence, Set, Callable

from torch import Tensor
from torch.nn import Module

from elasticai.creator.tags_utils import has_tag, get_tags, TagWrapper, tag

_precomputable_tag = 'precomputable'

WindowLength1D = int
WindowWidth = int
WindowHeight = int
InputChannels = int
BatchSize = int
Shape = tuple[int]
InputShape1D = tuple[InputChannels, WindowLength1D]
InputShape2D = tuple[InputChannels, WindowHeight, WindowWidth]
NestedTuple = Union[tuple['float', ...], tuple['NestedTuple', ...]]
CoefficientSet = Union[Tensor, Set[float]]


class Precomputation:
    def __init__(self, module: Module,
                 input_domain: Union[Callable[[], Tensor], Tensor]) -> None:
        self.module = module
        self.input_domain = input_domain
        self.output = None

    def _evaluate_input_domain_if_necessary(self) -> None:
        if isinstance(self.input_domain, Callable):
            self.input_domain = self.input_domain()

    @staticmethod
    def from_precomputable_tagged(module: TagWrapper[Module]):
        if not has_tag(module, _precomputable_tag):
            raise TypeError
        info_for_precomputation = get_tags(module)[_precomputable_tag]
        return Precomputation(module=module.unwrap(),
                              **info_for_precomputation)

    def __call__(self) -> None:
        """Precompute the input output pairs for the block handed during construction of the object.

        Be aware that this sets the corresponding submodule/block to eval mode. To continue training be sure to set
        it back to training mode again.
        """
        self._evaluate_input_domain_if_necessary()
        self.module.eval()
        self.output = self.module(self.input_domain)

    def __getitem__(self, item) -> Tensor:
        if item == 0:
            return self.input_domain
        elif item == 1:
            return self.output
        else:
            raise IndexError


def get_precomputations_from_direct_children(module):
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