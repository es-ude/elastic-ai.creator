from dataclasses import dataclass

import elasticai.creator.plugin as _pl
from elasticai.creator.ir import LoweringPass
from elasticai.creator.ir2vhdl import Implementation


class _TimeMultiplexedPass(LoweringPass[Implementation, Implementation]):
    pass


lower: LoweringPass[Implementation, Implementation] = _TimeMultiplexedPass()


@lower.register
def lutron(impl: Implementation) -> Implementation:
    return impl


def load_plugins(*packages: str) -> None:
    @dataclass
    class Spec(_pl.PluginSpec):
        generated: tuple[str, ...]

    def fetch(spec: Spec) -> _pl.PluginSymbol:
        return _pl.import_symbols(f"{spec.package}.src", spec.generated)

    fetcher = _pl.SymbolFetcherBuilder(Spec).add_fn(fetch).build()
    loader = _pl.PluginLoader(fetch=fetcher, plugin_receiver=lower)
    for p in packages:
        loader.load_from_package("elasticai.creator_plugins.{}".format(p))
