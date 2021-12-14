import unittest
from io import BytesIO

import onnx
import torch

from elasticai.creator.onnx import ModuleWrapper


class OnnxExportTest(unittest.TestCase):
    """
    TODO:
      - get rid of the warnings: The shape inference of custom_ops::Wrapper type is missing, so it may result in wrong
        shape inference for the exported graph. Please consider adding it in symbolic function.
    """

    def test_can_export_and_load(self):
        module = torch.nn.Sigmoid()
        module = ModuleWrapper(module)
        expected = self.get_simple_string_representation(operation_name="Wrapper", domain="custom_ops")

        with BytesIO() as buffer:
            torch.onnx.export(module, torch.ones(1, ), buffer)
            buffer.seek(0)
            model = onnx.load(buffer)
        self.assertEqual(expected, "{}".format(model))

    @staticmethod
    def get_simple_string_representation(operation_name, domain) -> str:
        template = """ir_version: 7
producer_name: "pytorch"
producer_version: "1.10"
graph {{
  node {{
    input: "input"
    output: "1"
    name: "{operation_name}_0"
    op_type: "{operation_name}"
    domain: "{domain}"
  }}
  name: "torch-jit-export"
  input {{
    name: "input"
    type {{
      tensor_type {{
        elem_type: 1
        shape {{
          dim {{
            dim_value: 1
          }}
        }}
      }}
    }}
  }}
  output {{
    name: "1"
    type {{
      tensor_type {{
        elem_type: 1
        shape {{
          dim {{
            dim_param: "{operation_name}1_dim_0"
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
  domain: "{domain}"
  version: 1
}}
"""
        template = template.format(domain=domain, operation_name=operation_name)
        return template


if __name__ == '__main__':
    unittest.main()
