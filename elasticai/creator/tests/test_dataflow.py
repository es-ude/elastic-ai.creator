from typing import Union, Iterable

from elasticai.creator.dataflow import DataFlowSpecification, DataSource
import unittest

from elasticai.creator.protocols import Tensor, Index, Module


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

    def __getitem__(self, index: Index) -> Tensor:
        return self


class DummyModule(Module):
    @property
    def training(self) -> bool:
        return True

    def extra_repr(self) -> str:
        return ""

    def named_children(self) -> Iterable[tuple[str, "Module"]]:
        yield from []

    def __call__(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x


class TestDataFlowSpecification(unittest.TestCase):
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
