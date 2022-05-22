import unittest
from abc import abstractmethod
from typing import Protocol, runtime_checkable
from unittest import TestCase

from elasticai.creator.metaprogramming import add_instance_attribute, add_method


class TestWrappingClasses(TestCase):
    @unittest.SkipTest
    def test_method(self):
        def my_method(self) -> list[int]:
            return [1]

        @add_method(method_name="my_method", method=my_method)
        class A:
            ...

        @runtime_checkable
        class P(Protocol):
            @abstractmethod
            def my_method(self) -> list[int]:
                ...

        # below assignment should be correct for static type analysis
        a: P = A()

        self.assertTrue(isinstance(a, P))

    def test_attribute(self):
        @add_instance_attribute(attribute_name="my_attribute", attribute=4)
        class A:
            ...

        self.assertTrue(not hasattr(A, "my_attribute"))
        self.assertEqual(4, A().my_attribute)
