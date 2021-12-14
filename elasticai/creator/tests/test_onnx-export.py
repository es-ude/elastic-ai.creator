import unittest
import torch
from elasticai.creator.precomputation import precomputable,Precomputation
import numpy
import onnx
from elasticai.creator.onnx_export import onnx_export_manager

class Onnxexport_test(unittest.TestCase):
    """
    in a model
    
    """
    def test_export_simple_sigmoid_precomputation(self):
        module = torch.nn.Sigmoid()
        precomputable_block =Precomputation(module,torch.Tensor(0,1))        
        onnx_export_manager.export_onnx(precomputable_block, (1,1), "sig.onnx")
        model =onnx.load("sig.onnx")
    
    def test_export_sigmoid_precomputation_check_properties(self):
        module = torch.nn.Sigmoid()
        precomputable_block =Precomputation(module,input_domain=torch.tensor([-1,1,0]))
        onnx_export_manager.export_onnx(precomputable_block, (1,1), "sig.onnx")
        model =onnx.load("sig.onnx")
        print(model)


if __name__ == '__main__':
    unittest.main()
