from elasticai.creator.tests.tensor_test_case import TensorTestCase

class DummyModule:
    def __init__(self, *children):
        self._children = children

    def children(self):
        yield from self._children


class MarkerTest(TensorTestCase):
    def test_precompute_block(self):
        module = DummyModule()
        module = tag_precomputed(module)
        self.assertIn("precomputed", module.elasticai_tags)


