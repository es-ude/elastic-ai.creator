from elasticai.creator.ir import Registry

from ._ir import (
    DataGraph,
    pattern_rule,
)
from ._ir import (
    sequential_with_interface as _sequential_with_interface,
)


def _replacement_fn(g: DataGraph, registry: Registry) -> tuple[DataGraph, Registry]:
    return _sequential_with_interface(("activation", "binarize")), registry


binarize_activations = pattern_rule(
    graph=_sequential_with_interface(("activation", "prelu")),
    replacement_fn=_replacement_fn,
)
