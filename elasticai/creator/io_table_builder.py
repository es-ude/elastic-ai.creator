from typing import Union, Tuple, List, Dict


import numpy as np
class Io_table_builder:
    
    @classmethod   
    def group_tables(cls, inputs: np.ndarray,outputs: np.ndarray, groups:int):
        """given an input, output pair return a list of arrays describing the io Tables of each group.
            Args:
                inputs: np.ndarray the inputs for the tables 
                outputs: np.ndarray the outputs for the tables
                groups: int the number of groups that will separate the said tables
            Returns:
              np.ndarray: The inputs
        """
        
        assert inputs.shape[1]%groups == 0 & outputs.shape[1]%groups == 0, "the first dimension of the arrays should be divisible"
        inputs,outputs = inputs,outputs
        input_io_dim = list(inputs.shape)
        output_io_dim= list(outputs.shape)
        input_io_dim[1] = input_io_dim[1]//groups
        output_io_dim[1] = output_io_dim[1]// groups
        io_tables = [(np.zeros(input_io_dim),np.zeros(output_io_dim)) for _ in range(groups)]
        for num,(input,output) in enumerate(zip(inputs,outputs)):
            for group in range(groups):
                input_length = input.shape[0]//groups
                output_length = output.shape[0]//groups
                io_tables[group][0][num] = input[group*input_length:group*input_length + input_length]
                io_tables[group][1][num] = output[group*output_length:group*output_length + output_length]
        return io_tables
    @classmethod
    def io_tables_to_dict(cls,tables: Union[Tuple[np.ndarray,np.ndarray],List[Tuple[np.ndarray,np.ndarray]]]) -> List[Dict]:
        """given a list or single input,output pair will return a list of dictionaries for each io pair. Said tables will be flatenned
            Args:
                tables: io tables in the format
            Returns:
              dict: A list of dictionaries for the io_tables both ends will be flatenned and transformed into tuples to facilitate iterating
        """
        dic_tables = []
        if not isinstance(tables,list):
            tables = [tables]
        for table_pairs in tables:
            dic_table = {}
            for input,output in zip(table_pairs[0],table_pairs[1]):
                dic_table[tuple(input.flatten().tolist())] = tuple(output.flatten().tolist() )
            dic_tables.append(dic_table)
        return  dic_tables

            
    