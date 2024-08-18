import torch


class MathOperations:
    def __init__(self):
        self.device = "cpu"

    def _clamp_result(
        self, result: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        result.clamp_(-(2 ** (c_quant_bits - 1)), (2 ** (c_quant_bits - 1)) - 1)
        return result

    def intadd(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        a = a.to(self.device)
        b = b.to(self.device)
        c = a + b
        return self._clamp_result(c, c_quant_bits)

    def intsub(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        a = a.to(self.device)
        b = b.to(self.device)
        c = a - b
        return self._clamp_result(c, c_quant_bits)

    def intmatmul(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        a = a.to(self.device)
        b = b.to(self.device)
        c = a.matmul(b)
        return self._clamp_result(c, c_quant_bits)
