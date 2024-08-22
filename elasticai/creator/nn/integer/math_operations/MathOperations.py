import torch


class MathOperations:
    def _clamp_result(
        self, result: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        result.clamp_(-(2 ** (c_quant_bits - 1)), (2 ** (c_quant_bits - 1)) - 1)
        return result

    def intadd(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        c = a + b.to(a.device)
        return self._clamp_result(c, c_quant_bits)

    def intsub(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        c = a - b.to(a.device)
        return self._clamp_result(c, c_quant_bits)

    def intmatmul(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        c = a.matmul(b.to(a.device))
        return self._clamp_result(c, c_quant_bits)
