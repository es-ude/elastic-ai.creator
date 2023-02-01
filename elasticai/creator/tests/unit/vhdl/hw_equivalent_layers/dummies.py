from typing import Any, Collection, Iterable, Iterator

from elasticai.creator.mlframework import Module
from elasticai.creator.mlframework.typing import Parameter
from elasticai.creator.vhdl.code import (
    Code,
    CodeFile,
    CodeModule,
    CodeModuleBase,
    Translatable,
)
from elasticai.creator.vhdl.hw_equivalent_layers.typing import HWEquivalentLayer


# noinspection PyMethodMayBeStatic
class DummyModule(HWEquivalentLayer):
    def signal_definitions(self, prefix: str) -> Code:
        return []

    def instantiation(self, prefix: str) -> Code:
        return []

    def translate(self) -> CodeModule:
        return CodeModuleBase(name="", files=[])

    @property
    def training(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return ""

    def named_children(self) -> Iterable[tuple[str, "Module"]]:
        yield from []

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from []

    def state_dict(self) -> dict[str, Any]:
        return {}

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        return x


class DummyCodeModule(CodeModule):
    @property
    def files(self) -> Collection[CodeFile]:
        return []

    @property
    def submodules(self) -> Collection["CodeModule"]:
        return []

    @property
    def name(self) -> str:
        return ""


class TranslatableDummyModule(DummyModule, Translatable):
    def translate(self) -> "CodeModule":
        return DummyCodeModule()
