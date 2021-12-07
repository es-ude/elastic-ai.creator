import unittest
import numpy as np
import torch
from elasticai.creator.derive_data_sets import create_input_data, find_unique_elements, create_io_table


def PyTorchAdapter():
    return torch


class derive_data_sets_test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ml_lib = PyTorchAdapter()

    def test_create_input_data_1d(self) -> None:
        table = create_input_data(input_dim=[2], domain=[0, 1])
        self.assertTrue((table == [[0, 0], [0, 1], [1, 0], [1, 1]]).all())

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
        self.assertRaises(SystemExit, create_input_data, [6, 6], [-1, 1])

    def test_create_io_table_basic(self) -> None:
        inputs = np.array([[1], [2], [3]])
        outputs = [self.ml_lib.tensor([2]), self.ml_lib.tensor([3]), self.ml_lib.tensor([4])]
        dict = create_io_table(inputs, outputs, channel_wise=False)
        self.assertTrue((dict == {(1,): (2,), (2,): (3,), (3,): (4,)}))

    def test_create_io_table_array(self) -> None:
        inputs = np.array([[1, 1], [2, 2], [3, 3]])
        outputs = [self.ml_lib.tensor([2, 2]), self.ml_lib.tensor([3, 3]), self.ml_lib.tensor([4, 4])]
        dict = create_io_table(inputs, outputs, channel_wise=False)
        self.assertTrue((dict == {(1, 1): (2, 2), (2, 2): (3, 3), (3, 3): (4, 4)}))

    def test_create_io_table_complex(self) -> None:
        inputs = np.array([[[1, 1], [1, 1]], [[2, 2], [2, 2]], [[3, 3], [3, 3]]])
        outputs = [self.ml_lib.tensor([2, 2]), self.ml_lib.tensor([3, 3]), self.ml_lib.tensor([4, 4])]
        dict = create_io_table(inputs, outputs, channel_wise=False)
        self.assertTrue((dict == {(1, 1, 1, 1): (2, 2), (2, 2, 2, 2): (3, 3), (3, 3, 3, 3): (4, 4)}))

    def test_create_io_table_channel_wise_basic(self) -> None:
        inputs = np.transpose(np.array([[[1], [2]], [[3], [4]]]),(0,2,1))
        outputs = [self.ml_lib.tensor([[[5], [6]]]), self.ml_lib.tensor([[[7], [8]]])]
        io_table = create_io_table(inputs, outputs, channel_wise=True)
        self.assertTrue((io_table == [{(1,): (5,), (3,): (7,)}, {(2,): (6,), (4,): (8,)}]))

    def test_create_io_table_channel_wise_array(self) -> None:
        inputs = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        outputs = [self.ml_lib.tensor([[2, 3], [4, 5]]), self.ml_lib.tensor([[6, 7], [8, 9]])]
        io_table = create_io_table(inputs, outputs, channel_wise=True)
        self.assertTrue((io_table == [{(1, 3): (2, 4), (5, 7): (6, 8)}, {(2, 4): (3, 5), (6, 8): (7, 9)}]))

    def test_find_unique_elements_basic(self) -> None:
            arr = find_unique_elements(np.array([0, 0, 1, 2]))
            self.assertTrue((arr == [0, 1, 2]).all())

    def test_find_unique_elements_complex(self) -> None:
        arr = find_unique_elements(np.array([[[1, 1], [2, 1]], [[2, 1], [1, 1]], [[1, 1], [2, 1]]]))
        self.assertTrue((list(arr.shape) == [2, 2, 2]))
        self.assertTrue((np.ndarray.flatten(arr) == np.ndarray.flatten(np.array([[[1, 1], [2, 1]], [[2, 1], [1, 1]]]))).all())


if __name__ == '__main__':
    unittest.main()
