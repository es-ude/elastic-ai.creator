import unittest
from io import BytesIO

import onnx
import torch

from elasticai.creator.onnx import ModuleWrapper
from elasticai.creator.tags_utils import tag
from elasticai.creator.tests.tensor_test_case import TensorTestCase

ONNX_HEADER = """ir_version: 4
producer_name: "pytorch"
producer_version: "1.11.0"
"""


class OnnxExportTest(TensorTestCase):
    """
    For documentation of the model returned by the onnx load function refer to
    https://github.com/onnx/onnx/blob/master/docs/IR.md

    TODO:
      - get rid of the warnings: The shape inference of elasticai.creator::Wrapper type is missing, so it may result in
        wrong shape inference for the exported graph. Please consider adding it in symbolic function.
    """

    def test_can_export_and_load(self):
        module = torch.nn.Sigmoid()
        module = ModuleWrapper(module)
        expected = self.get_simple_string_representation(
            operation_name=type(module.module).__name__, domain="elasticai.creator"
        )
        with BytesIO() as buffer:
            torch.onnx.export(
                module,
                torch.ones(
                    1,
                ),
                buffer,
            )
            buffer.seek(0)
            model = onnx.load(buffer)

        self.assertEqual(expected, "{}".format(model))

    @classmethod
    def get_simple_string_representation(cls, operation_name, domain) -> str:
        template = (
            ONNX_HEADER
            + """graph {
  node {
    output: "1"
    name: "Wrapper_0"
    op_type: "Wrapper"
"""
        )
        attributes = cls.build_attributes(
            """name: "operation_name"
      s: "{operation_name}"
      type: STRING""".format(
                operation_name=operation_name
            )
        )
        template = (
            template
            + attributes
            + """    domain: "elasticai.creator"
  }
  name: "torch-jit-export"
  output {
    name: "1"
    type {
      tensor_type {
        elem_type: 1
        shape {
          dim {
            dim_param: "Wrapper1_dim_0"
          }
        }
      }
    }
  }
}
opset_import {
  version: 9
}
opset_import {
  domain: "elasticai.creator"
  version: 1
}
"""
        )
        return template

    def test_with_tag_int(self):
        model = torch.nn.Sigmoid()
        input_shape = [3, 3]
        model = ModuleWrapper(tag(model, input_shape=input_shape))
        expected = self.get_string_representation_with_tag(
            operation_name=type(model.module).__name__,
            input_shape="""      ints: 3
      ints: 3
      type: INTS""",
        )

        self.check_stringified_onnx_model(expected_string=expected, model=model)

    def test_with_tag_float(self):
        model = torch.nn.Sigmoid()
        input_shape = [3.5, 3.5]
        model = ModuleWrapper(tag(model, input_shape=input_shape))
        expected = self.get_string_representation_with_tag(
            operation_name="Sigmoid",
            input_shape="""      floats: 3.5
      floats: 3.5
      type: FLOATS""",
        )

        self.check_stringified_onnx_model(expected_string=expected, model=model)

    def test_with_tag_Tensor(self):
        model = torch.nn.Sigmoid()
        input_shape = torch.tensor([1])
        model = ModuleWrapper(tag(model, input_shape=input_shape))
        expected = self.get_string_representation_with_tag(
            operation_name=type(model.module).__name__,
            input_shape="""      t {
        dims: 1
        data_type: 7
        raw_data: "\\001\\000\\000\\000\\000\\000\\000\\000"
      }
      type: TENSOR""",
        )

        self.check_stringified_onnx_model(expected_string=expected, model=model)

    def test_with_tag_String(self):
        model = torch.nn.Sigmoid()
        input_shape = "abc"
        model = ModuleWrapper(tag(model, input_shape=input_shape))
        expected = self.get_string_representation_with_tag(
            operation_name=type(model.module).__name__,
            input_shape="""      s: "abc"
      type: STRING""",
        )

        self.check_stringified_onnx_model(expected_string=expected, model=model)

    def check_stringified_onnx_model(self, expected_string, model):
        with BytesIO() as buffer:
            torch.onnx.export(
                model,
                torch.ones(
                    1,
                ),
                buffer,
            )
            buffer.seek(0)
            model = onnx.load(buffer)
        self.assertEqual(expected_string, "{}".format(model))

    @staticmethod
    def get_string_representation_with_tag(operation_name, input_shape) -> str:
        template = (
            ONNX_HEADER
            + """graph {{
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
    domain: "elasticai.creator"
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
  domain: "elasticai.creator"
  version: 1
}}
"""
        )
        template = template.format(
            operation_name=operation_name, input_shape=input_shape
        )
        return template

    @staticmethod
    def build_attributes(*args: str) -> str:
        attribute_string = ""
        for attribute in args:
            attribute_string += "    attribute {{\n      {attribute}\n    }}\n".format(
                attribute=attribute
            )
        return attribute_string

    def test_wrapper_transparent(self):
        module = torch.nn.Sigmoid()
        module = ModuleWrapper(module)
        input = torch.rand([2, 2])
        expected = module.module(input)
        actual = module(input)
        self.assertTensorEquals(expected, actual)


if __name__ == "__main__":
    unittest.main()
