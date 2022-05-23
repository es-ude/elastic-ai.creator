from types import SimpleNamespace

from elasticai.creator.tags_utils import get_tags, tag
from elasticai.creator.tests.tensor_test_case import TensorTestCase


class TagTest(TensorTestCase):
    def test_precompute_block(self):
        module = SimpleNamespace()
        module = tag(module, precomputed=None)
        self.assertIn("precomputed", module.elasticai_tags.keys())

    def test_precomputed_blocks_children_are_accessible(self):
        def children_generator():
            yield from [1, 2]

        module = SimpleNamespace(children=children_generator)
        module = tag(module)
        self.assertSequenceEqual([1, 2], list(module.children()))

    def test_precomputed_blocks_other_attributes_are_accessible(self):
        module = SimpleNamespace(other_attribute="some other attribute")
        module = tag(module)
        self.assertEqual("some other attribute", module.other_attribute)

    def test_update_tags(self):
        module = SimpleNamespace()
        module = tag(module, precomputed=None)
        module = tag(module, input_tensor_shape=(1, 2, 3))
        self.assertEqual(
            {
                "precomputed": None,
                "input_tensor_shape": (1, 2, 3),
            },
            get_tags(module),
        )
