import unittest
from itertools import repeat
from typing import Union, Iterable

from elasticai.creator.dataflow import DataSource, DataSink, sinks_have_common_source, group_dependent_sinks, \
    represent_grouped_DataFlowSpecification
from elasticai.creator.protocols import Tensor, Indices, Module, TensorMapping


class DummyTensor(Tensor):
    def as_strided(
        self, size: Union[tuple[int, ...], int], stride: Union[tuple[int, ...], int]
    ) -> Tensor:
        return self

    def select(self, dim: int, index: int) -> Tensor:
        return self

    @property
    def shape(self) -> tuple[int, ...]:
        return (1, 2)

    def __getitem__(self, index: Indices) -> Tensor:
        return self


class DummyModule(Module):
    @property
    def training(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return ""

    def named_children(self) -> Iterable[tuple[str, "Module"]]:
        yield from []

    def __call__(self, *args, **kwargs) -> Tensor:
        return args[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TestDataFlowSpecification(unittest.TestCase):
    """
    Test list:
    - test subsource of for selections
      - (2, 1), (1, 1)
      - (1, 1), (slice(0, 2), 1)
      - for slices of type
        - slice(0, None, None) x[0:]
        - slice(0, None, 2) x[0::2]
        - slice(None, -2, None) x[:-2]
        - slice(None, None, None) x[:]
      - define behaviour for tensor and numpy style slices
        - x[1, 2:4], which is i think basically equivalent to x[1][2:4], maybe seeing it as a tuple of slices helps

    """

    def test_source_unequals(self):
        module_a, module_b = self.create_dummy_modules(2)
        source_a, source_b = self.create_sources(
            modules=(module_a, module_b), selections=((1, 2), (1, 2))
        )
        self.assertFalse(source_a == source_b)

    def test_source_equals(self):
        (module,) = self.create_dummy_modules(1)
        source_a, source_b = self.create_sources(
            modules=repeat(module, 2), selections=repeat((1, 2), 2)
        )
        self.assertTrue(source_a == source_b)

    def test_subsource_of(self):
        module = DummyModule()
        source_a, source_b = self.create_sources(
            modules=repeat(module, 2), selections=((1, 1), (1, slice(0, 2)))
        )
        self.assertTrue(source_a.subsource_of(source_b))

    def test_not_subsource_of(self):
        module = DummyModule()
        source_a, source_b = self.create_sources(
            modules=repeat(module, 2), selections=((1, 1), (1, 2))
        )
        self.assertFalse(source_a.subsource_of(source_b))

    def test_repr_for_datasource(self):
        module = DummyModule()
        source = DataSource(node=module, selection=(1, 1))
        expected = "DataSource(source=DummyModule(), selection=(1, 1))"
        self.assertEqual(expected, repr(source))
    
    
    def test_sink_has_not_common_sources(self):
        module = DummyModule()
        source_a, source_b = self.create_sources(
            modules=repeat(module, 2), selections=((1, 1), (1, 2))
        )
        data_sink_a,data_sink_b = self.create_sinks(shapes=repeat((1, 1),2),sources=[[source_a],[source_b]],nodes=[None,None])
        self.assertFalse(sinks_have_common_source(data_sink_a,data_sink_b))
    
    def test_sink_has_common_sources_longer_b(self):
        module = DummyModule()
        source_a, source_b = self.create_sources(
            modules=repeat(module, 2), selections=((1, 1), (1, 2))
        )
        data_sink_a,data_sink_b = self.create_sinks(shapes=repeat((1, 1),2),sources=[[source_a],[source_a,source_b]],nodes=[None,None])
        self.assertFalse(sinks_have_common_source(data_sink_a,data_sink_b))
    
    def test_sink_has_common_sources(self):
        module = DummyModule()
        source_a = self.create_sources(
            modules=(module,), selections=((1, 1),)
        )
        data_sink_a,data_sink_b = self.create_sinks(shapes=repeat((1, 1),2),sources=[[source_a],[source_a]],nodes=[None,None])
        self.assertTrue(sinks_have_common_source(data_sink_a,data_sink_b))
    
    def test_group_independent_sinks(self):
        module = DummyModule()
        source_a,source_b = self.create_sources(
            modules=(repeat(module, 2)), selections=((1, 1), (1, 2))
        )
        data_sink_a,data_sink_b = self.create_sinks(shapes=repeat((1, 1),2),sources=[[source_a],[source_b]],nodes=[None,None]) 
        expected = ((data_sink_a,),(data_sink_b,))
        self.assertSequenceEqual(expected,group_dependent_sinks((data_sink_a,data_sink_b)))
    
    def test_group_dependent_sinks(self):
        module = DummyModule()
        source_a = self.create_sources(
            modules=(module,), selections=((1, 1),)
        )
        data_sink_a,data_sink_b = self.create_sinks(shapes=repeat((1, 1),2),sources=[[source_a],[source_a]],nodes=[None,None]) 
        expected = ((data_sink_a,data_sink_b),)
        self.assertSequenceEqual(expected,group_dependent_sinks((data_sink_a,data_sink_b)))
    
    def test_group_dependent_sinks_2groups(self):
        module = DummyModule()
        source_a, source_b = self.create_sources(
            modules=(repeat(module, 2)), selections=((1, 1), (1, 2))
        )
        data_sink_a,data_sink_b,data_sink_c = self.create_sinks(shapes=repeat((1, 1),3),sources =[[source_a],[source_b],[source_a]], nodes = repeat(None,3))  
        expected = ((data_sink_a,data_sink_c), (data_sink_b,))
        actual = group_dependent_sinks((data_sink_a, data_sink_b,data_sink_c))
        self.assertSequenceEqual(expected,actual )
    
    def test_represent_grouped_DataFlowSpecification(self):
        module = DummyModule()
        source_a, source_b = self.create_sources(
            modules=(repeat(module, 2)), selections=((1, 1), (1, 2))
        )
        data_sink_a,data_sink_b,data_sink_c = self.create_sinks(shapes=repeat((1, 1),3),sources =[[source_a],[source_b],[source_a]], nodes = self.create_dummy_modules(3))  
        actual = represent_grouped_DataFlowSpecification((data_sink_a, data_sink_b,data_sink_c))
        self.assertEqual("[DataSource(source=DummyModule(), selection=(1, 1))] -> (DummyModule(), selection = 0), (DummyModule(), selection = 0)\n[DataSource(source=DummyModule(), selection=(1, 2))] -> (DummyModule(), selection = 0)\n",actual )
    
    @staticmethod
    def create_sinks(shapes: Iterable[tuple[int]],sources:Iterable[DataSource],nodes: Iterable[Union[TensorMapping,None]]):
        return tuple(DataSink(source,shape,node,0) for shape,source,node in zip(shapes,sources,nodes) )
    
    @staticmethod
    def create_dummy_modules(n: int) -> tuple[Module, ...]:
        return tuple((DummyModule() for _ in range(n)))
    @staticmethod
    def create_sources(
        modules: Iterable[TensorMapping],
        selections: Iterable[Union[int, slice, Indices]],
    ) -> tuple[DataSource, ...]:
        return tuple(
            (
                DataSource(node=source, selection=selection)
                for source, selection in zip(modules, selections)
            )
        )
