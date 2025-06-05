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

    def int_mac(
        self,
        x: torch.IntTensor,
        w: torch.IntTensor,
        x_quant_bits: int,
        w_quant_bits: int,
        b: torch.IntTensor = None,
    ) -> torch.IntTensor:
        assert x.dtype == torch.int32
        assert w.dtype == torch.int32

        tmp = F.linear(x, w)

        tmp_quant_bits = (x_quant_bits + 1) + (w_quant_bits + 1)
        self.clamp_result(tmp, tmp_quant_bits)

        if b is not None:
            assert b.dtype == torch.int32
            tmp = self.intadd(tmp, b, tmp_quant_bits + 1)

        return tmp

    def int_matmul_4d(
        self,
        inputs1: torch.IntTensor,
        inputs2: torch.IntTensor,
        operation_mode: str,
        outputs_quant_bits: int,
    ) -> torch.Tensor:
        assert inputs1.dtype == torch.int32
        assert inputs2.dtype == torch.int32

        if operation_mode == "score":
            inputs1 = inputs1.transpose(1, 2)
            inputs2 = inputs2.transpose(1, 2)
            inputs2_transpose = inputs2.transpose(-2, -1)
            outputs = inputs1.matmul(inputs2_transpose)  # QK^T

        elif operation_mode == "att":
            inputs2 = inputs2.transpose(1, 2)
            outputs = inputs1.matmul(inputs2).transpose(1, 2)  # softmax_out * V
        else:
            raise NotImplementedError

        return self.clamp_result(outputs, outputs_quant_bits)

    def matmul_4d(
        self,
        inputs1: torch.FloatTensor,
        inputs2: torch.FloatTensor,
        operation_mode: str,
    ) -> torch.Tensor:
        assert inputs1.dtype == torch.float32
        assert inputs2.dtype == torch.float32

        if operation_mode == "score":
            inputs1 = inputs1.transpose(1, 2)
            inputs2 = inputs2.transpose(1, 2)
            inputs2_transpose = inputs2.transpose(-2, -1)
            outputs = inputs1.matmul(inputs2_transpose)  # QK^T

        elif operation_mode == "att":
            inputs2 = inputs2.transpose(1, 2)
            outputs = inputs1.matmul(inputs2).transpose(1, 2)  # softmax_out * V
        else:
            raise NotImplementedError

        return outputs

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

    def int_dotproduct(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32
        assert b.dtype == torch.int32

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        dot_product = a_cpu * b_cpu

        dot_product = dot_product.to(a.device)
        return self.clamp_result(dot_product, c_quant_bits)

    def int_division(
        self, a: torch.IntTensor, b: torch.IntTensor, c_quant_bits: int
    ) -> torch.IntTensor:
        assert a.dtype == torch.int32 or a.dtype == torch.int64
        assert b.dtype == torch.int32 or b.dtype == torch.int64

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        division = a_cpu // b_cpu

        division = division.to(a.device)
        return self.clamp_result(division, c_quant_bits)
