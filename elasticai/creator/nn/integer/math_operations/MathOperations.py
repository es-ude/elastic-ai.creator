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
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32
        c = a + b.to(a.device)
        return self._clamp_result(c, c_quant_bits)

    def intsub(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32

        c = a - b.to(a.device)
        return self._clamp_result(c, c_quant_bits)

    def intmatmul(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        m, k = a_cpu.shape
        k, n = b_cpu.shape

        c_cpu = torch.zeros((m, n), dtype=torch.int32)

        for i in range(m):
            for j in range(n):
                for p in range(k):
                    c_cpu[i, j] += a_cpu[i, p].item() * b_cpu[p, j].item()
        c = c_cpu.to(a.device)
        return self._clamp_result(c, c_quant_bits)

        # c = a.matmul(b.to(a.device))
        # return self._clamp_result(c, c_quant_bits)
