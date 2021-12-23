import unittest

from elasticai.creator.brevitas.brevitas_model_comparison import (
    BrevitasModelComparisonTestCase,
)
from elasticai.creator.brevitas.brevitas_representation import (
    BrevitasRepresentation,
)
from elasticai.creator.systemTests.models_definition import (
    create_qtorch_model,
    create_brevitas_model,
    define_weight,
)


class ModelSystemTest(BrevitasModelComparisonTestCase):
    """
    System tests for translating a big qtorch model to brevitas
    """

    def test_complete_models_with_weights(self) -> None:
        self.qtorch_model = create_qtorch_model()

        self.brevitas_model = create_brevitas_model()
        define_weight([layer for layer in self.brevitas_model])

        translated_model = BrevitasRepresentation.from_pytorch(
            self.qtorch_model
        ).translated_model
        translated_layers = [layer for layer in translated_model]
        define_weight(translated_layers)

        self.assertModelEqual(translated_model, self.brevitas_model)


if __name__ == "__main__":
    unittest.main()
