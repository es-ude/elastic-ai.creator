from typing import Any, Iterable, Iterator

from elasticai.creator.mlframework import Module
from elasticai.creator.mlframework.typing import Parameter
from elasticai.creator.tests.unit.vhdl_for_deprecation.translator.pytorch.test_translator import (
    CodeModuleBase,
)
from elasticai.creator.vhdl.code import CodeModule
from elasticai.creator.vhdl.code.code import Code
from elasticai.creator.vhdl.translatable_modules.typing import HWEquivalentLayer


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
