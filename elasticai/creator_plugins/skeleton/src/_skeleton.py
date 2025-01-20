import importlib.resources as res

from elasticai.creator.ir2vhdl import Code, Implementation, Ir2Vhdl
from elasticai.creator.plugin import PluginSymbol as Loadable
from elasticai.creator.vhdl_template import (
    EntityTemplateDirector,
    Template,
)


def _load_vhd(component: str) -> str:
    vhd_file = res.files(".").joinpath(f"../vhd/{component}.vhd")
    return vhd_file.read_text()


class _Skeleton(Loadable[Ir2Vhdl]):
    def __init__(self, template: Template):
        self._template = template

    def __call__(self, arg: Implementation) -> Code:
        return "", [""]

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
        loader.register("skeleton")(skeleton)


skeleton = _Skeleton
