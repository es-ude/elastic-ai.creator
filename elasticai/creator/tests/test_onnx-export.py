import unittest
import torch
from elasticai.creator.precomputation import precomputable,Precomputation
import numpy
import onnx
from elasticai.creator.onnx_export import onnx_export_manager
import os
class Onnxexport_test(unittest.TestCase):
    def test_export_simple_sigmoid_precomputation(self):
        module = torch.nn.Sigmoid()
        precomputable_block =Precomputation(module,torch.Tensor(0,1),input_shape=[1,2])        
        onnx_export_manager.export_onnx(precomputable_block, (1,1), "sig.onnx")
        model =onnx.load("sig.onnx")
    
    def test_export_sigmoid_precomputation_check_properties(self):
        module = torch.nn.Sigmoid()
        input_shape = [1,2]
        precomputable_block =Precomputation(module,input_domain=torch.tensor([-1,1,0]),input_shape=input_shape)
        onnx_export_manager.export_onnx(precomputable_block, (1,1), "sig.onnx")
        model =onnx.load("sig.onnx")
        nodes = model.graph.node
        for node in nodes:
            if node.op_type == "Precomputation" :
                x = node.attribute
                input = node.attribute[0].ints
        self.assertSequenceEqual(input,input_shape)
    
    
    def tearDown(self) -> None:
        os.remove("sig.onnx")
    


if __name__ == '__main__':
    unittest.main()
