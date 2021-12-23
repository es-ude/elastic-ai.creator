import unittest
import torch

from elasticai.creator.systemTests.models_definition import (
    create_qtorch_model,
    create_brevitas_model,
    define_weight,
)

# for reproducability
torch.manual_seed(0)


class InputSystemTest(unittest.TestCase):
    """
    System test where the same input is put in a qtorch model and the corresponding tranlsated brevitas model
    Check if the output is the same
    """

    def test_input(self) -> None:
        self.qtorch_model = create_qtorch_model()
        self.brevitas_model = create_brevitas_model()
        define_weight([layer for layer in self.qtorch_model])
        define_weight([layer for layer in self.brevitas_model])

        output_qtorch = self.qtorch_model(torch.ones(1, 1, 100))
        output_brevitas = self.brevitas_model(torch.ones(1, 1, 100))

        self.assertTrue(torch.equal(output_qtorch, output_brevitas))

    def test_bigger_input(self) -> None:
        self.qtorch_model = create_qtorch_model()
        self.brevitas_model = create_brevitas_model()
        define_weight([layer for layer in self.qtorch_model])
        define_weight([layer for layer in self.brevitas_model])

        output_qtorch = self.qtorch_model(torch.ones(512, 1, 100))
        output_brevitas = self.brevitas_model(torch.ones(512, 1, 100))

        self.assertTrue(torch.equal(output_qtorch, output_brevitas))


if __name__ == "__main__":
    unittest.main()
