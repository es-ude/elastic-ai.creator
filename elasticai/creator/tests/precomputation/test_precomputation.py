import json
from collections import Iterable

import numpy as np
import torch

from elasticai.creator.precomputation import JSONEncoder, Precomputation, precomputable
from elasticai.creator.tags_utils import Module, get_tags
from elasticai.creator.tests.tensor_test_case import TensorTestCase


class DummyModule:
    def __init__(self):
        self.call = lambda x: x

    @property
    def training(self) -> bool:
        return False

    # noinspection PyMethodMayBeStatic
    def extra_repr(self) -> str:
        return ""

    # noinspection PyMethodMayBeStatic
    def named_children(self) -> Iterable[tuple[str, "Module"]]:
        yield from ()

    def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.call(x)


class PrecomputationTest(TensorTestCase):
    def test_precompute(self):
        module = DummyModule()

        def call(x: torch.Tensor) -> torch.Tensor:
            return x * 2

        module.call = call
        precompute = Precomputation(
            module=module, input_domain=torch.tensor([[1], [1]])
        )
        precompute()
        self.assertTensorEquals(torch.tensor([[2], [2]]), precompute.output)

    def test_precomputation_is_json_encodable(self):
        module = DummyModule()
        precompute = Precomputation(module=module, input_domain=torch.tensor([[]]))
        precompute()
        actual = json.dumps(precompute, cls=JSONEncoder)
        expected = """{"description": [], "shape": [0], "x": [[]], "y": [[]]}"""
        self.assertEqual(expected, actual)

    def test_precomputation_is_json_encodable_for_shape1x1(self):
        module = DummyModule()
        precompute = Precomputation(module=module, input_domain=torch.tensor([[[1]]]))
        precompute()
        actual = json.dumps(precompute, cls=JSONEncoder)
        expected = (
            """{"description": [], "shape": [1, 1], "x": [[[1]]], "y": [[[1]]]}"""
        )
        self.assertEqual(expected, actual)

    def test_precomputation_is_json_encoding_correctly_writes_in_and_output(self):
        module = DummyModule()

        def call(x: torch.Tensor) -> torch.Tensor:
            return torch.flatten(x * 3)

        module.call = call
        precompute = Precomputation(module=module, input_domain=torch.tensor([[[1]]]))
        precompute()
        actual = json.dumps(precompute, cls=JSONEncoder)
        expected = """{"description": [], "shape": [1, 1], "x": [[[1]]], "y": [3]}"""
        self.assertEqual(expected, actual)

    def test_precomputable_tag(self):
        input_generator = [-1, 1]

        @precomputable(input_shape=[-1, 1], input_generator=input_generator)
        class MyCustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

        module = MyCustomModule()
        tags = get_tags(module)

        self.assertSequenceEqual(tags["precomputable"]["input_shape"], [-1, 1])
