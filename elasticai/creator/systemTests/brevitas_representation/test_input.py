import random
import unittest

import numpy as np
import torch

from elasticai.creator.systemTests.brevitas_representation.models_definition import (
    create_brevitas_model,
    create_qtorch_model,
)


class InputSystemTest(unittest.TestCase):
    """
    System test where the same input is put in a qtorch model and the corresponding tranlsated brevitas model
    Check if the output is the same
    """

    def setUp(self) -> None:
        self.ensure_reproducibility()

    @staticmethod
    def ensure_reproducibility():
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def test_input(self) -> None:
        self.qtorch_model = create_qtorch_model()
        # we think brevitas is manipulating some seed therefore we need to reset them again
        self.ensure_reproducibility()
        self.brevitas_model = create_brevitas_model()

        output_qtorch = self.qtorch_model(torch.ones(1, 1, 100))
        output_brevitas = self.brevitas_model(torch.ones(1, 1, 100))

        self.assertTrue(torch.equal(output_qtorch, output_brevitas))

    def test_bigger_input(self) -> None:
        self.qtorch_model = create_qtorch_model()
        # we think brevitas is manipulating some seed therefore we need to reset them again
        self.ensure_reproducibility()
        self.brevitas_model = create_brevitas_model()

        output_qtorch = self.qtorch_model(torch.ones(512, 1, 100))
        output_brevitas = self.brevitas_model(torch.ones(512, 1, 100))

        self.assertTrue(torch.equal(output_qtorch, output_brevitas))


if __name__ == "__main__":
    unittest.main()
