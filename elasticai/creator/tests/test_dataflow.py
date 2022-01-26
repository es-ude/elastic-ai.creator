import unittest
from typing import Union, Iterable

from elasticai.creator.dataflow import DataSource
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


class DummyTensorMapping(TensorMapping):
    def __repr__(self) -> str:
        pass

    def __call__(self, *args, **kwargs) -> Tensor:
        pass


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
        module_a = DummyModule()
        module_b = DummyModule()
        source_a = DataSource(source=module_a, selection=(1, 2))
        source_b = DataSource(source=module_b, selection=(1, 2))
        self.assertFalse(source_a == source_b)

    def test_source_equals(self):
        module = DummyModule()
        source_a = DataSource(source=module, selection=(1, 2))
        source_b = DataSource(source=module, selection=(1, 2))
        self.assertTrue(source_a == source_b)

    def test_subsource_of(self):
        module = DummyModule()
        source_a = DataSource(source=module, selection=(1, 1))
        source_b = DataSource(source=module, selection=(1, slice(0, 2)))
        self.assertTrue(source_a.subsource_of(source_b))

    def test_not_subsource_of(self):
        module = DummyModule()
        source_a = DataSource(source=module, selection=(1, 1))
        source_b = DataSource(source=module, selection=(1, 2))
        self.assertFalse(source_a.subsource_of(source_b))

    def test_repr_for_datasource(self):
        module = DummyModule()
        source = DataSource(source=module, selection=(1, 1))
        expected = "DataSource(source=DummyModule(), selection=(1, 1))"
        self.assertEqual(expected, repr(source))
