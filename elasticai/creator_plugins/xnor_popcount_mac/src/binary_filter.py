from string import Template

from elasticai.creator import ir2vhdl as ir
from elasticai.creator.file_generation.resource_utils import get_file_from_package


@ir.type_handler()
def binary_filter(original: ir.DataGraph, registry: ir.Registry) -> ir.Code:
    global _template
    if _template is None:
        _template = _prepare_template()
    kernel_size = original.attributes["kernel_size"]
    weight = original.attributes["weight"]
    weight_bits = _normalize_weight_bits(weight)
    if kernel_size <= 0:
        raise ValueError(f"kernel_size must be > 0, got {kernel_size}")
    if len(weight_bits) % kernel_size != 0:
        raise ValueError(
            "weight bit length must be divisible by kernel_size, got "
            f"len(weight_bits)={len(weight_bits)} and kernel_size={kernel_size}"
        )
    num_channels = len(weight_bits) // kernel_size
    return original.name, [
        _template.substitute(
            KERNEL_SIZE=original.attributes["kernel_size"],
            NUM_OUT_CHANNELS=num_channels,
            PARALLEL_INSTANCES=original.attributes["parallelism"],
            weight=weight,
            entity=original.name,
        )
    ]


def _normalize_weight_bits(weight: str) -> str:
    if len(weight) >= 2 and weight[0] == '"' and weight[-1] == '"':
        return weight[1:-1]
    return weight


def _prepare_template():
    with _load_prototype() as prototype_resource:
        with open(prototype_resource, "r") as f:
            prototype = f.read()
    director = ir.EntityTemplateDirector().set_prototype(prototype)
    for name in ("KERNEL_SIZE", "NUM_OUT_CHANNELS", "PARALLEL_INSTANCES", "weight"):
        director.add_value(name)

    return director.build()


def _load_prototype():
    return get_file_from_package(
        "elasticai.creator_plugins.xnor_popcount_mac", "vhdl/binary_filter.vhd"
    )


_template: None | Template = None
