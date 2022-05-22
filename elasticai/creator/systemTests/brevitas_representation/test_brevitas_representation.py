import random
import unittest

import numpy as np
import torch

from elasticai.creator.brevitas.brevitas_model_comparison import (
    BrevitasModelComparisonTestCase,
)
from elasticai.creator.brevitas.brevitas_representation import BrevitasRepresentation
from elasticai.creator.systemTests.brevitas_representation.models_definition import (
    create_brevitas_model,
    create_qtorch_model,
)


class ModelSystemTest(BrevitasModelComparisonTestCase):
    """
    System tests for translating a big qtorch model to brevitas
    """

    def setUp(self) -> None:
        self.ensure_reproducibility()

    @staticmethod
    def ensure_reproducibility():
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)

    def test_complete_models_with_weights(self) -> None:
        self.qtorch_model = create_qtorch_model()
        # we think brevitas is manipulating some seed therefore we need to reset them again
        self.ensure_reproducibility()
        self.brevitas_model = create_brevitas_model()
        # we think brevitas is manipulating some seed therefore we need to reset them again
        self.ensure_reproducibility()

        translated_model = BrevitasRepresentation.from_pytorch(
            self.qtorch_model
        ).translated_model

        self.assertModelEqual(translated_model, self.brevitas_model)


if __name__ == "__main__":
    unittest.main()
