import unittest

import torch

from elasticai.creator.precomputation import create_input_tensor
from elasticai.creator.tests.tensor_test_case import TensorTestCase


class PrecomputationTest(TensorTestCase):
    @unittest.SkipTest
    def test_create_input_tensor(self):
        result = create_input_tensor(
            input_shape=(1, 2),
            input_domain=(-1, 1),
            groups=1
        )
        expected = torch.tensor(
            (((-1, -1),),
             ((-1, 1),),
             ((1, -1),),
             ((1, 1),))
        )
        self.assertTensorEquals(expected, result)
