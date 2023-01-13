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
from elasticai.creator.vhdl.model_tracing import (
    HWEquivalentGraph,
    HWEquivalentNode,
    Node,
)


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


class DummyModuleNode(Node):
    @property
    def op(self) -> str:
        return "call_module"

    def __init__(self, name=""):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class DummyHWEquivalentNode(DummyModuleNode, HWEquivalentNode):
    def __init__(self, name=""):
        super().__init__(name)
        self._module = DummyModule()

    @property
    def hw_equivalent_layer(self) -> HWEquivalentLayer:
        return self._module

    @hw_equivalent_layer.setter
    def hw_equivalent_layer(self, value):
        self._module = value


class DummyGraph(HWEquivalentGraph):
    @property
    def hw_equivalent_nodes(self) -> Iterable[HWEquivalentNode]:
        yield from []

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def nodes(self) -> Iterable[HWEquivalentNode]:
        return self._nodes


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
