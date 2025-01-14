import torch
import torch.nn.functional as F


class MathOperations:
    def clamp_result(self, result: torch.IntTensor, quant_bits: int) -> torch.IntTensor:
        result.clamp_(-(2 ** (quant_bits - 1)), (2 ** (quant_bits - 1)) - 1)
        return result

    def intadd(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32
        c = a + b.to(a.device)

        return self.clamp_result(c, c_quant_bits)

    def intsub(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32

        c = a - b.to(a.device)
        return self.clamp_result(c, c_quant_bits)

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
        return self.clamp_result(c, c_quant_bits)

    def intconv1d(
        self,
        x: torch.IntTensor,
        w: torch.IntTensor,
        b: torch.IntTensor,
        padding: tuple[int, int] or str,
        tmp_quant_bits: int,
    ) -> torch.IntTensor:
        assert x.dtype == torch.int32
        assert w.dtype == torch.int32
        assert b.dtype == torch.int32

        if b is not None:
            assert b.dtype == torch.int32
            tmp = F.conv1d(x, w, b, padding=padding)
        else:
            tmp = F.conv1d(x, w, padding=padding)
        return self.clamp_result(tmp, tmp_quant_bits)

    def inthadamardproduct(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32
        assert a.shape == b.shape

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        dot_product = a_cpu * b_cpu

        dot_product = dot_product.to(a.device)
        return self.clamp_result(dot_product, c_quant_bits)
