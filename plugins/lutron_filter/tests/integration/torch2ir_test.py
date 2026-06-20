import torch
import torch.nn as tnn
from elasticai.creator_plugins.lutron_filter import get_default_torch2ir
from torch.testing import assert_close as assert_tensors_close

torch2ir = get_default_torch2ir()


class TestConv1d:
    def get_conv_and_ir(self):
        conv = tnn.Conv1d(in_channels=2, out_channels=2, kernel_size=3, bias=True)
        model = tnn.Sequential(conv)
        ir = torch2ir(model)
        ir_conv = ir[1]["0"]
        return conv, ir_conv

    def test_kernel_is_stored(self):
        conv, ir_conv = self.get_conv_and_ir()
        ir_conv_kernel = ir_conv.attributes.get_mapping("parameters").get_tuple(
            "weight"
        )
        ir_conv_kernel_tensor = torch.tensor(ir_conv_kernel)
        assert_tensors_close(conv.weight, ir_conv_kernel_tensor)

    def test_bias_is_stored(self):
        conv, ir_conv = self.get_conv_and_ir()
        ir_bias = ir_conv.attributes.get_mapping("parameters").get_tuple("bias")
        ir_bias_tensor = torch.tensor(ir_bias)
        assert_tensors_close(ir_bias_tensor, conv.bias)
