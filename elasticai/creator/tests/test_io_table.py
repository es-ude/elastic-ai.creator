import unittest
from elasticai.creator.io_table import IOTable, group_tables
import torch


class test_Io_table_builder(unittest.TestCase):
    def test_group_tables_basic(self):
        inputs = torch.Tensor([[1, 2], [3, 4]])
        outputs = torch.Tensor([[4, 5], [5, 4]])
        Iotable = IOTable(inputs, outputs)
        result = group_tables(Iotable, groups=1)
        self.assertTrue(
            torch.all(inputs == result[0].tables[0]),
            torch.all(outputs == result[0].tables[1]),
        )
    def test_group_tables_depthwise(self):
        inputs = torch.Tensor(
            [
                [
                    1,
                    2,
                ],
                [3, 4],
            ]
        )
        outputs = torch.Tensor([[1, 2], [6, 5]])
        Iotable = IOTable(inputs, outputs)
        result = group_tables(Iotable, groups=2)
        expected = [[], []]
        expected[0] = (torch.Tensor([[1], [3]]), torch.Tensor([[1], [6]]))
        expected[1] = (torch.Tensor([[2], [4]]), torch.Tensor([[2], [5]]))
        for expected_pair, result_pair in zip(expected, result):
            self.assertTrue(torch.all(expected_pair[0] == result_pair.inputs))
            self.assertTrue(torch.all(expected_pair[1] == result_pair.outputs))

    def test_group_tables_depthwise_longer_input(self):
        inputs = torch.Tensor([[1, 2, 3, 4], [3, 4, 5, 6]])
        outputs = torch.Tensor([[1, 2], [6, 5]])
        Iotable = IOTable(inputs, outputs)
        result = group_tables(Iotable, groups=2)
        expected = [[], []]
        expected[0] = (torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1], [6]]))
        expected[1] = (torch.Tensor([[3, 4], [5, 6]]), torch.Tensor([[2], [5]]))
        for expected_pair, result_pair in zip(expected, result):
            self.assertTrue(torch.all(expected_pair[0] == result_pair.inputs))
            self.assertTrue(torch.all(expected_pair[1] == result_pair.outputs))

    def test_create_io_dict_basic(self) -> None:
        inputs = torch.tensor([[1], [2], [3]])
        outputs = torch.tensor([[2], [3], [4]])
        Iotable = IOTable(inputs, outputs)
        dict = Iotable.get_table_as_dict()
        self.assertTrue((dict == {(1,): (2,), (2,): (3,), (3,): (4,)}))
        
    def test_io_repr_basic(self) -> None:
        inputs = torch.tensor([[1], [2], [3]])
        outputs = torch.tensor([[2], [3], [4]])
        Iotable = IOTable(inputs, outputs)
        repr = Iotable.__repr__()
        self.assertEqual(repr,"[1]:[2], [2]:[3], [3]:[4]")

    def test_create_io_dict_depthwise(self) -> None:
        io_list = [
            IOTable(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1], [6]])),
            IOTable(torch.Tensor([[3, 4], [5, 6]]), torch.Tensor([[2], [5]])),
        ]
        dict_list = io_list[0].get_table_as_dict(), io_list[1].get_table_as_dict()
        self.assertTrue((dict_list[0] == {(1, 2): (1,), (3, 4): (6,)}))
        self.assertTrue((dict_list[1] == {(3, 4): (2,), (5, 6): (5,)}))
        
    
    def test_select_tables_output_indices(self):
        inputs = torch.Tensor([[1, 2], [3, 4]])
        outputs = torch.Tensor([[0,0,1], [0,0,0]])
        Iotable = IOTable(inputs, outputs)
        Iotable.select_outputs(2)
        self.assertTrue(
            torch.all(torch.Tensor([1,0]) == Iotable.outputs).item()
        )


if __name__ == "__main__":
    unittest.main()
