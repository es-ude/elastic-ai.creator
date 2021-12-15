import unittest
from io import BytesIO

import onnx
import torch

from elasticai.creator.onnx import ModuleWrapper
from elasticai.creator.tags_utils import tag

class OnnxExportTest(unittest.TestCase):


    def test_can_export_and_load(self):
        module = torch.nn.Sigmoid()
        module = ModuleWrapper(module)
        expected = self.get_simple_string_representation(operation_name=type(module.module).__name__, domain="custom_ops")

        with BytesIO() as buffer:
            torch.onnx.export(module, torch.ones(1, ), buffer)
            buffer.seek(0)
            model = onnx.load(buffer)
        self.assertEqual(expected, "{}".format(model))\
    
    @staticmethod
    def get_simple_string_representation(operation_name, domain) -> str:
        template = """ir_version: 7
producer_name: "pytorch"
producer_version: "1.10"
graph {{
  node {{
    output: "1"
    name: "Wrapper_0"
    op_type: "Wrapper"
    attribute {{
      name: "operation_name"
      s: "{operation_name}"
      type: STRING
    }}
    domain: "custom_ops"
  }}
  name: "torch-jit-export"
  output {{
    name: "1"
    type {{
      tensor_type {{
        elem_type: 1
        shape {{
          dim {{
            dim_param: "Wrapper1_dim_0"
          }}
        }}
      }}
    }}
  }}
}}
opset_import {{
  version: 9
}}
opset_import {{
  domain: "custom_ops"
  version: 1
}}
"""
        template = template.format(operation_name=operation_name)
        return template
    
    def get_buffer__from_model(self,module):
        buffer = BytesIO()
        torch.onnx.export(module, torch.ones(1, ), buffer)
        buffer.seek(0)
        model = onnx.load(buffer)
        return model
    
    def test_with_tag_int(self):
        module = torch.nn.Sigmoid()
        input_shape = [3,3]
        module = ModuleWrapper(tag(module,input_shape=input_shape))
        expected = self.get_string_representation_with_tag(operation_name=type(module.module).__name__, domain="custom_ops",input_shape="""      ints: 3
      ints: 3
      type: INTS""")
        
        model = self.get_buffer__from_model(module)
        self.assertEqual(expected, "{}".format(model))


    def test_with_tag_float(self):
        module = torch.nn.Sigmoid()
        input_shape = [3.5, 3.5]
        module = ModuleWrapper(tag(module, input_shape=input_shape))
        expected = self.get_string_representation_with_tag(operation_name=type(module.module).__name__,
                                                           domain="custom_ops", input_shape="""      floats: 3.5
      floats: 3.5
      type: FLOATS""")

        model = self.get_buffer__from_model(module)
        self.assertEqual(expected, "{}".format(model))

    def test_with_tag_strings(self):
        module = torch.nn.Sigmoid()
        input_shape = torch.tensor([1])
        module = ModuleWrapper(tag(module, input_shape=input_shape))
        expected = self.get_string_representation_with_tag(operation_name=type(module.module).__name__,
                                                           domain="custom_ops", input_shape="""      t {
        dims: 1
        data_type: 7
        raw_data: "\\001\\000\\000\\000\\000\\000\\000\\000"
      }
      type: TENSOR""")

        model = self.get_buffer__from_model(module)
        self.assertEqual(expected, "{}".format(model))
        
   
    @staticmethod
    def get_string_representation_with_tag(operation_name, domain,input_shape) -> str:
        template = """ir_version: 7
producer_name: "pytorch"
producer_version: "1.10"
graph {{
  node {{
    output: "1"
    name: "Wrapper_0"
    op_type: "Wrapper"
    attribute {{
      name: "input_shape"
{input_shape}
    }}
    attribute {{
      name: "operation_name"
      s: "{operation_name}"
      type: STRING
    }}
    domain: "custom_ops"
  }}
  name: "torch-jit-export"
  output {{
    name: "1"
    type {{
      tensor_type {{
        elem_type: 1
        shape {{
          dim {{
            dim_param: "Wrapper1_dim_0"
          }}
        }}
      }}
    }}
  }}
}}
opset_import {{
  version: 9
}}
opset_import {{
  domain: "custom_ops"
  version: 1
}}
"""
        template = template.format(operation_name=operation_name,input_shape=input_shape)
        return template


    def test_wrapper_transparent(self):
        module = torch.nn.Sigmoid()
        module = ModuleWrapper(module)
        input = torch.rand([2,2])
        expected =  module.module(input)
        actual = module(input)
        self.assertTrue(torch.all(torch.eq(expected, actual)))

if __name__ == '__main__':
    unittest.main()
