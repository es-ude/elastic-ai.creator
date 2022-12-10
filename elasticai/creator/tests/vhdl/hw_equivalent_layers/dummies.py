from typing import Iterable, Iterator, Any, Collection

from mlframework import Module
from mlframework.typing import Parameter
from vhdl.code import Code, CodeModule, CodeFile, Translatable
from vhdl.hw_equivalent_layers.typing import HWEquivalentLayer
from vhdl.model_tracing import HWEquivalentNode, Node, HWEquivalentGraph


class DummyModule(HWEquivalentLayer):
    def signals(self, prefix: str) -> Code:
        yield from []

    def instantiation(self, prefix: str) -> Code:
        yield from []

    def translate(self) -> "CodeModule":
        pass

    @property
    def data_width(self) -> int:
        return 1

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
        pass


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
