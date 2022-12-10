from typing import Iterable, Any, Iterator, Collection
from unittest import TestCase

from mlframework import Module
from mlframework.typing import Parameter
from vhdl.code import Translatable, CodeModule, CodeFile
from vhdl.graph_parsing import collect_modules
from vhdl.model_tracing import Graph, ModuleNode, Node


class DummyModule(Module):
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


class DummyModuleNode(ModuleNode):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name=""):
        self._name = name
        self._module = DummyModule()

    @property
    def module(self) -> Module:
        return self._module

    @module.setter
    def module(self, val):
        self._module = val


class DummyNode(Node):
    def __init__(self, name=""):
        self._name = name

    @property
    def name(self) -> str:
        return self._name


class DummyGraph(Graph):
    @property
    def module_nodes(self) -> Iterable[ModuleNode]:
        yield from []

    def __init__(self, nodes):
        self._nodes = nodes

    @property
    def nodes(self) -> Iterable[ModuleNode]:
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


class CollectUniqueModulesFromGraph(TestCase):
    def test_collects_all_unique_module_instances_for_two_unique_inputs(self):
        node_a = DummyModuleNode()
        node_b = DummyModuleNode()
        g = DummyGraph((node_b, node_a))
        modules = collect_modules(g)
        self.assertEqual({node_b.module, node_a.module}, modules)

    def test_collects_all_unique_instances_for_two_identical_inputs(self):
        node_a = DummyModuleNode()
        node_b = DummyModuleNode()
        node_a.module = node_b.module
        g = DummyGraph((node_b, node_a))
        modules = collect_modules(g)
        self.assertEqual({node_a.module}, modules)

    def test_collects_instances_for_three_unique_modules(self):
        nodes = tuple(DummyModuleNode() for _ in range(3))
        expected = set((n.module for n in nodes))
        g = DummyGraph(nodes)
        modules = collect_modules(g)
        self.assertEqual(expected, modules)

    def test_collects_only_instances_from_module_nodes(self):
        nodes = (DummyModuleNode(), DummyNode())
        g = DummyGraph(nodes)
        expected = {nodes[0].module}
        self.assertEqual(expected, collect_modules(g))
