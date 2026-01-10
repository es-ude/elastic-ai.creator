import importlib.resources as res
from string import Template

import elasticai.creator.ir.ir_v2 as ir
from elasticai.creator.ir2vhdl import (
    Code,
    DataGraph,
    EntityTemplateDirector,
    Ir2Vhdl,
    PluginSymbol,
)


def _load_vhd(component: str) -> str:
    vhd_file = res.files(package="elasticai.creator_plugins.skeleton").joinpath(
        f"vhdl/{component}.vhd"
    )
    return vhd_file.read_text()


class _Skeleton(PluginSymbol):
    def __init__(self, template: Template):
        self._template = template

    def __call__(self, arg: DataGraph, registry: ir.Registry) -> Code:
        return arg.name, self._template.substitute(
            arg.attributes["generic_map"] | dict(entity=arg.name)
        )

    @classmethod
    def load_into(cls, loader: Ir2Vhdl) -> None:
        template = (
            EntityTemplateDirector()
            .set_prototype(_load_vhd("skeleton"))
            .add_generic("DATA_IN_WIDTH")
            .add_generic("DATA_IN_DEPTH")
            .add_generic("DATA_OUT_WIDTH")
            .add_generic("DATA_OUT_DEPTH")
            .build()
        )
        skeleton = cls(template)
        loader.register("skeleton", skeleton)


skeleton = _Skeleton
