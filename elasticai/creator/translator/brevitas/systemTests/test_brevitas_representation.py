import unittest

from elasticai.creator.translator.brevitas.brevitas_model_comparison import BrevitasModelComparisonTestCase
from elasticai.creator.translator.brevitas.brevitas_representation import BrevitasRepresentation
from elasticai.creator.translator.brevitas.systemTests.models_definition import create_qtorch_model, crete_brevitas_model, define_weight


class ModelSystemTest(BrevitasModelComparisonTestCase):
    """
    System tests for translating a big qtorch model to brevitas
    """

    def test_complete_models_with_weights(self) -> None:
        self.qtorch_model = create_qtorch_model()

        self.brevitas_model = crete_brevitas_model()
        define_weight([layer for layer in self.brevitas_model])

        translated_model = BrevitasRepresentation.from_pytorch(self.qtorch_model).translated_model
        translated_layers = [layer for layer in translated_model]
        define_weight(translated_layers)

        self.assertModelEqual(translated_model, self.brevitas_model)


if __name__ == "__main__":
    unittest.main()
