"""
Test list:
- make sure that the create_input_data functions from elasticai.creator.derive_datasets
  are used correctly
- make sure precomputations are serializable and loadable
- depthwise convolutions are handled correctly

Extra:
- compare given input_domains shape and rank to the input_shape and reuse and
  use input_domain's elements to build a set of tensors of shape input_shape. E.g.:
  input_shape = (2, 2)
  input_domain = [(1, 1), (-1, 1)]
  expected_result = [((1, 1), (1, 1)),
                     ((1, 1), (-1, 1)),
                     ((-1, 1), (1, 1)),
                     ((-1, 1), (-1, 1))]
"""
import json
from collections import Iterable

import numpy as np
import torch

from elasticai.creator.precomputation import Precomputation, JSONEncoder, ModuleProto
from elasticai.creator.tests.tensor_test_case import TensorTestCase


class PrecomputationTest(TensorTestCase):
    def test_precomputing_a_function(self):
        def function(x):
            return "{} more".format(x)

        def noop_function():
            pass

        function.eval = noop_function
        precompute = Precomputation(module=function, input_domain="something")
        precompute()
        self.assertEqual(("something", "something more"), tuple(precompute))

    def test_precomputation_is_json_encodable(self):

        class DummyModule(ModuleProto):
            @property
            def training(self) -> bool:
                return False

            def extra_repr(self) -> str:
                return ""

            def named_children(self) -> Iterable[str, 'ModuleProto']:
                raise IndexError

            def __call__(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
                return x

        module = DummyModule()
        precompute = Precomputation(module=module, input_domain=np.ndarray([[1, 1]]))
        json_string = json.dumps(precompute, cls=JSONEncoder)
