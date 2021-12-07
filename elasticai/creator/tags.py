from elasticai.creator.tags_utils import tag, TagWrapper
from typing import TypeVar, Tuple, Sequence, Union
from torch.nn import Module

T = TypeVar('T')


def precomputable(module: Union[Module, TagWrapper[Module]],
                  input_shape: Tuple[int],
                  groups: int,
                  input_domain: Sequence[float]) -> TagWrapper[Module]:
    """Add all necessary information to allow later tools to precompute the specified module

    The arguments provided will be used to determine the input data that needs
    to be fed to the module to produce the precomputed input-output table.
    """
    return tag(module, precomputed={
        'input_shape': input_shape,
        'input_domain': input_domain,
        'groups': groups,
    })
