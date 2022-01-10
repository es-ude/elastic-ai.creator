from typing import Union, Tuple, List, Dict

import torch
import numpy as np
class Io_table_builder:
    
    @classmethod   
    def group_tables(cls,inputs: torch.Tensor,outputs: torch.Tensor, groups:int, groups_dim = 1):
        inputs,outputs = inputs,outputs
        input_io_dim = inputs.shape
        output_io_dim= outputs.shape
        input_io_dim[groups_dim] = input_io_dim[groups_dim]//groups
        output_io_dim[groups_dim] = output_io_dim[groups_dim]// groups
        io_tables = [(torch.zeros(input_io_dim),torch.zeros(output_io_dim)) for _ in groups]
        for num,input,output in enumerate(zip(inputs,outputs)):
            for group in range(groups):
                input_length = input.shape[groups_dim]//groups
                output_length = output.shape[groups_dim]//groups
                io_tables[group][0][num] = input[group*input_length:group*input_length + input_length]
                io_tables[group][1][num] = output[group*output_length:group*output_length + output_length]
    @classmethod
    def io_tables_to_dict(cls,tables: Union[Tuple[torch.Tensor,torch.Tensor],]) -> List[Dict]:
        dic_tables = []
        if not isinstance(tables,List):
            tables = [tables]
        for table_pairs in tables:
            dic_table = {}
            for input,output in zip(table_pairs[0],table_pairs[1]):
                dic_table[input.numpy().flatten().tolist()] = output.numpy().flatten().tolist() 
            dic_tables.append(dic_table)
        return  dic_tables

            
    