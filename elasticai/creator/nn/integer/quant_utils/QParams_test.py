# # TODO：to test updateScaleZeropoint() -> effect the updateScaleZeropoint() in the Linear.forward()
# #

# import unittest

# import torch

# class TestQuantizeProcess(unittest.TestCase):
#     def test_quantizeProcess_with_AS_signed(self):
#         x = torch.tensor([-2.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32).to("cpu")
#         min_quant = torch.tensor([-128], dtype=torch.int32).to("cpu")
#         max_quant = torch.tensor([127], dtype=torch.int32).to("cpu")
#         scale_factor = torch.tensor([0.0196078431372549], dtype=torch.float32).to("cpu")
#         zero_point = torch.tensor([-26], dtype=torch.int32).to("cpu")

#         results = quantizeProcess(x, min_quant, max_quant, scale_factor, zero_point)
#         expected = torch.tensor([-128, 0, 25, 76, 127], dtype=torch.int32)
#         self.assertTrue(torch.equal(results, expected))

#     def test_quantizeProcess_with_AS_unsigned(self):
#         x = torch.tensor([-2.0, 0.5, 1.0, 2.0, 3.0], dtype=torch.float32).to("cpu")
#         min_quant = torch.tensor([0], dtype=torch.int32).to("cpu")
#         max_quant = torch.tensor([255], dtype=torch.int32).to("cpu")
#         scale_factor = torch.tensor([0.0196078431372549], dtype=torch.float32).to("cpu")
#         zero_point = torch.tensor([102], dtype=torch.int32).to("cpu")

#         results = quantizeProcess(x, min_quant, max_quant, scale_factor, zero_point)
#         expected = torch.tensor([0, 128, 153, 204, 255], dtype=torch.int32)
#         self.assertTrue(torch.equal(results, expected))

#     def test_quantizeProcess_with_SS_signed(self):
#         x = torch.tensor([-2.0, 0.5, 0.5, 1.0, 2.0], dtype=torch.float32).to("cpu")
#         min_quant = torch.tensor([-127], dtype=torch.int32).to("cpu")
#         max_quant = torch.tensor([127], dtype=torch.int32).to("cpu")
#         scale_factor = torch.tensor([0.01968503937007874], dtype=torch.float32).to(
#             "cpu"
#         )
#         zero_point = torch.tensor([-25], dtype=torch.int32).to("cpu")

#         results = quantizeProcess(x, min_quant, max_quant, scale_factor, zero_point)
#         expected = torch.tensor([-127, 0, 0, 26, 77], dtype=torch.int32)
#         self.assertTrue(torch.equal(results, expected))

#     def test_quantizeProcess_with_SS_unsigned(self):
#         x = torch.tensor([-2.0, 0.5, 0.5, 1.0, 2.0], dtype=torch.float32).to("cpu")
#         min_quant = torch.tensor([0], dtype=torch.int32).to("cpu")
#         max_quant = torch.tensor([254], dtype=torch.int32).to("cpu")
#         scale_factor = torch.tensor([0.01968503937007874], dtype=torch.float32).to(
#             "cpu"
#         )
#         zero_point = torch.tensor([101], dtype=torch.int32).to("cpu")

#         results = quantizeProcess(x, min_quant, max_quant, scale_factor, zero_point)
#         expected = torch.tensor([0, 126, 126, 152, 203], dtype=torch.int32)
#         self.assertTrue(torch.equal(results, expected))
