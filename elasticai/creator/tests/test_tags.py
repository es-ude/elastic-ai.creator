from types import SimpleNamespace

from elasticai.creator.tests.tensor_test_case import TensorTestCase
from elasticai.creator.tags import tag_precomputed, tag


class TagTest(TensorTestCase):
    def test_precompute_block(self):
        module = SimpleNamespace()
        module = tag_precomputed(module)
        self.assertIn("precomputed", module.tags().keys())

    def test_precomputed_blocks_children_are_accessible(self):
        def children_generator():
            yield from [1, 2]

        module = SimpleNamespace(children=children_generator)
        module = tag_precomputed(module)
        self.assertSequenceEqual([1, 2], list(module.children()))

    def test_precomputed_blocks_other_attributes_are_accessible(self):
        module = SimpleNamespace(other_attribute="some other attribute")
        module = tag_precomputed(module)
        self.assertEqual("some other attribute", module.other_attribute)

    def test_wrapped_object_is_not_wrapped_again(self):
        module = SimpleNamespace(children=[1, 2])
        module = tag_precomputed(module)
        module = tag_precomputed(module)
        self.assertFalse(hasattr(module.unwrap(), "unwrap"))

    def test_update_tags(self):
        module = SimpleNamespace()
        module = tag_precomputed(module)
        module = tag(module, input_tensor_shape=(1, 2, 3))
        self.assertEquals({
            "precomputed": None,
            "input_tensor_shape": (1, 2, 3),
        }, module.tags())
