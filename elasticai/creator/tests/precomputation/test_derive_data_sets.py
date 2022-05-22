import itertools
import unittest

import numpy as np
import torch
from numpy.testing import assert_equal as assertNPArrayEqual

from elasticai.creator.input_domains import (
    construct_codomain_from_elements,
    create_input_data,
    create_io_table,
    find_unique_elements,
    get_cartesian_product_from_items,
)


def PyTorchAdapter():
    return torch


class DeriveDataSetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ml_lib = PyTorchAdapter()

    def test_create_input_data_1d(self) -> None:
        table = create_input_data(input_dim=[2], domain=[0, 1])
        self.assertTrue(table.tolist(), [[0, 0], [0, 1], [1, 0], [1, 1]])

    def test_create_input_data_none(self) -> None:
        table = create_input_data([2, None], [0, 1])
        self.assertTrue((table == [[0, 0], [0, 1], [1, 0], [1, 1]]).all())

    def test_create_input_data_2d(self) -> None:
        table = create_input_data([1, 2], [0, 1])
        self.assertTrue((table == [[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]]).all())

    def test_create_input_data_ternary(self) -> None:
        table = create_input_data([1], [0, 1, -1])
        self.assertTrue((table == [[-1], [0], [1]]).all())

    def test_create_input_data_raiseSystemExit(self) -> None:
        self.assertRaises(MemoryError, create_input_data, [6, 6], [-1, 1])

    def test_create_io_table_basic(self) -> None:
        inputs = np.array([[1], [2], [3]])
        outputs = [
            self.ml_lib.tensor([2]),
            self.ml_lib.tensor([3]),
            self.ml_lib.tensor([4]),
        ]
        dict = create_io_table(inputs, outputs, channel_wise=False)
        self.assertTrue((dict == {(1,): (2,), (2,): (3,), (3,): (4,)}))

    def test_create_io_table_array(self) -> None:
        inputs = np.array([[1, 1], [2, 2], [3, 3]])
        outputs = [
            self.ml_lib.tensor([2, 2]),
            self.ml_lib.tensor([3, 3]),
            self.ml_lib.tensor([4, 4]),
        ]
        dict = create_io_table(inputs, outputs, channel_wise=False)
        self.assertTrue((dict == {(1, 1): (2, 2), (2, 2): (3, 3), (3, 3): (4, 4)}))

    def test_create_io_table_complex(self) -> None:
        inputs = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])
        outputs = [
            self.ml_lib.tensor([2, 2]),
            self.ml_lib.tensor([3, 3]),
            self.ml_lib.tensor([4, 4]),
        ]
        dict = create_io_table(inputs, outputs, channel_wise=False)
        self.assertTrue(
            (dict == {(1, 1, 1, 1): (2, 2), (2, 2, 2, 2): (3, 3), (3, 3, 3, 3): (4, 4)})
        )

    def test_create_io_table_channel_wise_basic(self) -> None:
        inputs = np.transpose(np.array([[[1], [2]], [[3], [4]]]), (0, 2, 1))
        outputs = [self.ml_lib.tensor([[[5], [6]]]), self.ml_lib.tensor([[[7], [8]]])]
        io_table = create_io_table(inputs, outputs, channel_wise=True)
        self.assertTrue(
            (io_table == [{(1,): (5,), (3,): (7,)}, {(2,): (6,), (4,): (8,)}])
        )

    def test_create_io_table_channel_wise_array(self) -> None:
        inputs = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        outputs = [
            self.ml_lib.tensor([[2, 3], [4, 5]]),
            self.ml_lib.tensor([[6, 7], [8, 9]]),
        ]
        io_table = create_io_table(inputs, outputs, channel_wise=True)
        self.assertTrue(
            (
                io_table
                == [{(1, 3): (2, 4), (5, 7): (6, 8)}, {(2, 4): (3, 5), (6, 8): (7, 9)}]
            )
        )

    def test_find_unique_elements_basic(self) -> None:
        arr = find_unique_elements(np.array([0, 0, 1, 2]))
        self.assertTrue((arr == [0, 1, 2]).all())

    def test_find_unique_elements_complex(self) -> None:
        arr = find_unique_elements(
            np.array([[[1, 1], [2, 1]], [[2, 1], [1, 1]], [[1, 1], [2, 1]]])
        )
        self.assertTrue((list(arr.shape) == [2, 2, 2]))
        self.assertTrue(
            (
                np.ndarray.flatten(arr)
                == np.ndarray.flatten(np.array([[[1, 1], [2, 1]], [[2, 1], [1, 1]]]))
            ).all()
        )

    def test_cartesian_product_of_one_element_is_a_repetition(self) -> None:
        expected = np.array(((1, 1),), dtype="float16")
        actual = get_cartesian_product_from_items(length=2, items=[1])
        assertNPArrayEqual(actual, expected)

    def test_cartesian_product_of_one_element_is_repeated_three_times(self) -> None:
        expected = np.array(((1, 1, 1),), dtype="float16")
        actual = get_cartesian_product_from_items(length=3, items=[1])
        assertNPArrayEqual(actual, expected)

    def test_cartesian_product_for_two_elements(self):
        expected = np.array(((1, 1), (1, 2), (2, 1), (2, 2)), dtype="float16")
        actual = get_cartesian_product_from_items(length=2, items=(1, 2))
        assertNPArrayEqual(actual, expected)

    def test_cartesian_product_for_two_vector_elements(self):
        expected = (
            ((0, 0), (0, 0)),
            ((0, 0), (1, 1)),
            ((1, 1), (0, 0)),
            ((1, 1), (1, 1)),
        )
        expected = np.array(expected, dtype="float16")
        actual = get_cartesian_product_from_items(length=2, items=((0, 0), (1, 1)))
        assertNPArrayEqual(expected, actual)

    def test_cartesian_nested(self):
        expected = itertools.product((0, 1), repeat=4)
        expected = ([[a, b], [c, d]] for a, b, c, d in expected)
        expected = np.array(list(expected), dtype="float16")
        actual = get_cartesian_product_from_items(items=(0, 1), length=2)
        actual = get_cartesian_product_from_items(items=actual, length=2)
        assertNPArrayEqual(expected, actual)

    def test_construct_domain_yields_domain_with_2_2_shaped_elements(self):
        expected = itertools.product((0, 1), repeat=4)
        expected = ([[a, b], [c, d]] for a, b, c, d in expected)
        expected = np.array(list(expected), dtype="float16")
        actual = construct_codomain_from_elements(
            shape=(2, 2), codomain_elements=(0, 1)
        )
        assertNPArrayEqual(expected, actual)

    def test_construct_domain_from_subshape_vectors_length_2(self):
        expected = itertools.product((0, 1), repeat=2)
        table = ((0, 0), (1, 1))
        expected = ((table[i], table[j]) for i, j in expected)
        expected = np.array(list(expected), dtype="float16")
        actual = construct_codomain_from_elements(
            shape=(2, 2), codomain_elements=((0, 0), (1, 1))
        )
        assertNPArrayEqual(expected, actual)

    def test_construct_domain_from_subshape_vectors_length_3(self):
        expected = itertools.product((0, 1), repeat=3)
        table = ((0, 0), (1, 1))
        expected = ((table[i], table[j], table[k]) for i, j, k in expected)
        expected = np.array(list(expected), dtype="float16")
        actual = construct_codomain_from_elements(
            shape=(3, 2), codomain_elements=((0, 0), (1, 1))
        )
        assertNPArrayEqual(expected, actual)

    def test_construct_domain_from_incompatible_subshape_raises_error(self):
        self.assertRaises(
            ValueError,
            construct_codomain_from_elements,
            shape=(2, 3),
            codomain_elements=((0, 0), (1, 1)),
        )


if __name__ == "__main__":
    unittest.main()
