import unittest
from elasticai.creator.io_table_builder import Io_table_builder
import  numpy as np

class test_Io_table_builder(unittest.TestCase):
    def test_group_tables_basic(self):
        inputs = np.asarray([[1,2],[3,4]])
        outputs = np.asarray([[4,5],[5,4]])
        result = Io_table_builder.group_tables(inputs,outputs,groups=1)
        self.assertTrue(np.all(inputs==result[0][0]),np.all(outputs==result[0][1]))
        
    def test_group_tables_depthwise(self):
        inputs = np.asarray([[1,2,],[3,4]])
        outputs = np.asarray([[1,2],[6,5]])
        result = Io_table_builder.group_tables(inputs, outputs, groups=2)
        expected = [[],[]]
        expected[0] = (np.asarray([[1],[3]]),np.asarray([[1], [6]]))
        expected[1] = (np.asarray([[2],[4]]),np.asarray([[2], [5]]))
        for expected_pair, result_pair in zip(expected,result):
            self.assertTrue(np.all(expected_pair[0]==result_pair[0]))
            self.assertTrue(np.all(expected_pair[1]==result_pair[1]))
    
    
    def test_group_tables_depthwise_longer_input(self):
        inputs = np.asarray([[1,2,3,4],[3,4,5,6]])
        outputs = np.asarray([[1,2],[6,5]])
        result = Io_table_builder.group_tables(inputs, outputs, groups=2)
        expected = [[],[]]
        expected[0] = (np.asarray([[1,2],[3,4]]),np.asarray([[1], [6]]))
        expected[1] = (np.asarray([[3,4],[5,6]]),np.asarray([[2], [5]]))
        for expected_pair, result_pair in zip(expected,result):
            self.assertTrue(np.all(expected_pair[0]==result_pair[0]))
            self.assertTrue(np.all(expected_pair[1]==result_pair[1]))
            
    def test_create_io_dict_basic(self) -> None:
        inputs = np.array([[1], [2], [3]])
        outputs = np.array([[2], [3], [4]])
        dict = Io_table_builder.io_tables_to_dict((inputs, outputs))[0]
        self.assertTrue((dict == {(1,): (2,), (2,): (3,), (3,): (4,)}))

    def test_create_io_dict_depthwise(self) -> None:
        io_list =  [(np.asarray([[1,2],[3,4]]),np.asarray([[1], [6]])),(np.asarray([[3,4],[5,6]]),np.asarray([[2], [5]]))] 
        dict = Io_table_builder.io_tables_to_dict(io_list)
        self.assertTrue((dict[0] == {(1,2): (1,), (3,4): (6,)}))
        self.assertTrue((dict[1] == {(3, 4): (2,), (5, 6): (5,)}))



if __name__ == '__main__':
    unittest.main()
