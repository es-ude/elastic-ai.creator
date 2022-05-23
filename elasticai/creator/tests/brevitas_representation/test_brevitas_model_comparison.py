import unittest

from elasticai.creator.brevitas.brevitas_model_comparison import BrevitasModelMatcher


def dummy_model():
    yield from []


class DummyLayer:
    def __init__(self, *parameters):
        self._parameters = [DummyParameter(x) for x in parameters]

    def parameters(self):
        yield from self._parameters


class DummyParameter:
    def __init__(self, value):
        self.value = value

    def equal(self, other):
        if other is None:
            return False
        return self.value == other.value


class BrevitasModelComparisonTest(unittest.TestCase):
    def test_one_model_and_None_raise_Typeerror(self):
        self.assertRaises(
            TypeError, BrevitasModelMatcher.check_equality, None, dummy_model()
        )
        self.assertRaises(
            TypeError, BrevitasModelMatcher.check_equality, dummy_model(), None
        )

    def test_empty_models_are_equal(self):
        fake_model_1 = dummy_model()
        fake_model_2 = dummy_model()
        equality = BrevitasModelMatcher.check_equality(fake_model_1, fake_model_2)
        self.assertTrue(equality)

    def test_models_with_one_layer_are_equal(self):
        fake_model_1 = [DummyLayer()]
        fake_model_2 = [DummyLayer()]
        equality = BrevitasModelMatcher.check_equality(fake_model_1, fake_model_2)
        self.assertTrue(equality)

    def test_models_with_one_layer_are_unequal(self):
        fake_model_1 = [1]
        fake_model_2 = ["bla"]
        equality = BrevitasModelMatcher.check_equality(fake_model_1, fake_model_2)
        self.assertFalse(equality)

    def test_models_with_different_number_of_layers(self):
        dummy_model_1 = [DummyLayer(), "bla"]
        dummy_model_2 = [DummyLayer()]
        equality = BrevitasModelMatcher.check_equality(dummy_model_1, dummy_model_2)
        self.assertFalse(equality)

    def test_non_iterable_models_throw_exception(self):
        dummy_model_1 = 1
        dummy_model_2 = 2
        self.assertRaises(
            TypeError, BrevitasModelMatcher.check_equality, dummy_model_1, dummy_model_2
        )

    def test_models_with_same_layer_types_but_different_parameters_are_unequal(self):
        dummy_model_1 = [DummyLayer(1)]
        dummy_model_2 = [DummyLayer()]
        equality = BrevitasModelMatcher.check_equality(dummy_model_1, dummy_model_2)
        self.assertFalse(equality)

    def test_models_with_same_layer_types_but_different_parameters_are_unequal_exchanged(
        self,
    ):
        dummy_model_1 = [DummyLayer()]
        dummy_model_2 = [DummyLayer(1)]
        equality = BrevitasModelMatcher.check_equality(dummy_model_1, dummy_model_2)
        self.assertFalse(equality)

    def test_same_models_with_completely_same_layers_are_equal(self):
        dummy_model_1 = [DummyLayer(1)]
        dummy_model_2 = [DummyLayer(1)]
        equality = BrevitasModelMatcher.check_equality(dummy_model_1, dummy_model_2)
        self.assertTrue(equality)


if __name__ == "__main__":
    unittest.main()
