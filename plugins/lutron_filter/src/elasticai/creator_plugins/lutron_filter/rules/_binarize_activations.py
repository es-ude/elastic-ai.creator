from elasticai.creator.ir import Registry, attribute

from ._ir import (
    DataGraph,
    pattern_rule,
)
from ._ir import (
    sequential_with_interface as _sequential_with_interface,
)


def _replacement_fn(g: DataGraph, registry: Registry) -> tuple[DataGraph, Registry]:
    new_g = _sequential_with_interface(("activation", "binarize"))
    new_bin = new_g.nodes["activation"]
    new_g = new_g.add_node(
        new_bin.name, new_bin.attributes | {"implementation": "binarize"}
    )
    new_reg = registry | {
        "binarize": g.clear().with_attributes(attribute(type="binarize"))
    }
    return new_g, new_reg


binarize_activations = pattern_rule(
    graph=_sequential_with_interface(("activation", "prelu")),
    replacement_fn=_replacement_fn,
)
