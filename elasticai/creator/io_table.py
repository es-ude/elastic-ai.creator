from typing import Dict, List, Tuple, Union

import numpy as np


class IOTable:
    def __init__(self, inputs: np.ndarray, outputs: np.ndarray):
        self.inputs = inputs
        self.outputs = outputs
        pass

    @property
    def tables(self):
        return self.inputs, self.outputs

    def __getitem__(self, item):
        return (self.inputs[item], self.outputs[item])

    def get_table_as_dict(self) -> Dict:
        """given a list or single input,output table pair will return a list of dictionaries for each io pair. Said tables will be flatenned.
        Args:
            tables: io tables in the format
        Returns:
          dict: A list of dictionaries for the io_tables both ends will be flattened and transformed into tuples to facilitate iterating.
        """

        dic_table = {}
        n = self.tables
        for input, output in self:
            dic_table[tuple(input.flatten().tolist())] = tuple(
                output.flatten().tolist()
            )
        return dic_table


def group_tables(table: IOTable, groups: int) -> List[IOTable]:
    """given an input, output pair return a list of arrays describing the io Tables of each group.
    Args:
        table: IOTable the target IOTable
        groups: int the number of groups that will separate the said tables, those groups will break the tables along the first non-batch dimension.
    Returns:
      List[IOTable]: The grouped tables as a list of input output objects
    """
    inputs = table.tables[0]
    outputs = table.tables[1]
    assert (
        inputs.shape[1] % groups == 0 & outputs.shape[1] % groups == 0
    ), "the first dimension of the arrays should be divisible by the number of groups"
    inputs, outputs = inputs, outputs
    input_io_dim = list(inputs.shape)
    output_io_dim = list(outputs.shape)
    input_io_dim[1] = input_io_dim[1] // groups
    output_io_dim[1] = output_io_dim[1] // groups
    io_tables = [
        (np.zeros(input_io_dim), np.zeros(output_io_dim)) for _ in range(groups)
    ]
    for num, (input, output) in enumerate(zip(inputs, outputs)):
        for group in range(groups):
            input_length = input.shape[0] // groups
            output_length = output.shape[0] // groups
            io_tables[group][0][num] = input[
                group * input_length : group * input_length + input_length
            ]
            io_tables[group][1][num] = output[
                group * output_length : group * output_length + output_length
            ]
    io_tables_obj = []
    for table in io_tables:
        io_tables_obj.append(IOTable(table[0], table[1]))

    return io_tables_obj
