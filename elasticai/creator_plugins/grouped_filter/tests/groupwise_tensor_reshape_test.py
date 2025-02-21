from unittest import TestCase

from elasticai.creator_plugins.grouped_filter.src.iterator_utils import batched
from elasticai.creator_plugins.grouped_filter.src.tensor import (
    reshape_flat_CxN_groupwise,
    structure_flat_CxN,
)


class ReshapeCxNTest(TestCase):
    def test_interpret_flat_vector_as_2x6(self):
        expected = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))
        self.assertEqual(expected, structure_flat_CxN(tuple(range(12)), channels=6))

    def test_interpret_flat_vector_as_3x4(self):
        expected = (
            (
                0,
                1,
                2,
            ),
            (3, 4, 5),
            (6, 7, 8),
            (9, 10, 11),
        )
        self.assertEqual(expected, structure_flat_CxN(tuple(range(12)), channels=3))


class BatchedTest(TestCase):
    def test_batched(self):
        expected = ([0, 1], [2, 3], [4, 5])
        actual = tuple(batched(list(range(6)), 2))
        self.assertEqual(expected, actual)


class GroupWiseConnectionTest(TestCase):
    def test_interpret_flat_vector_as_6x2_and_reshape_to_3_groups(self):
        expected = ((0, 1, 6, 7), (2, 3, 8, 9), (4, 5, 10, 11))

        self.assertEqual(
            expected, reshape_flat_CxN_groupwise(tuple(range(12)), channels=6, groups=3)
        )

    def test_interpret_flat_vector_as_4x3_and_reshape_to_4_groups(self):
        expected = ((0, 4, 8), (1, 5, 9), (2, 6, 10), (3, 7, 11))
        self.assertEqual(
            expected, reshape_flat_CxN_groupwise(tuple(range(12)), channels=4, groups=4)
        )

    def test_interpret_flat_vector_as_6x4_and_reshape_to_2_groups(self):
        expected = (
            (0, 1, 2, 6, 7, 8, 12, 13, 14, 18, 19, 20),
            (3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23),
        )
        self.assertEqual(
            expected, reshape_flat_CxN_groupwise(tuple(range(24)), channels=6, groups=2)
        )
