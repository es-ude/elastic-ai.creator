import importlib.resources as res
from string import Template

from elasticai.creator.ir2vhdl import (
    Code,
    EntityTemplateDirector,
    Implementation,
    Ir2Vhdl,
    PluginSymbol,
)

from ._skeleton import skeleton as skeleton


class _BufferedNetworkWrapper(PluginSymbol):
    def __init__(self, template: Template):
        self._template = template

    def __call__(self, args: Implementation) -> Code:
        def _iter():
            yield self._template.substitute(
                dict(entity=args.name) | args.attributes["generic_map"]
            )

        return args.name, _iter()

    @classmethod
    def load_into(cls, receiver: Ir2Vhdl) -> None:
        def load_file():
            return (
                res.files("elasticai.creator_plugins.skeleton")
                .joinpath("vhdl/buffered_network_wrapper.vhd")
                .read_text()
            )

        template = (
            EntityTemplateDirector()
            .set_prototype(load_file())
            .add_generic("KERNEL_SIZE")
            .add_generic("STRIDE")
        ).build()
        receiver.register("buffered_network_wrapper", cls(template))


buffered_network_wrapper = _BufferedNetworkWrapper
