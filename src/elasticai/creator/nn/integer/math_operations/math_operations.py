import torch
import torch.nn.functional as F

from elasticai.creator.nn.integer._dtypes import check_dtype


class MathOperations:
    """Integer arithmetic that mimics fixed-width hardware operations.

    All integer operations check their operand dtypes and clamp the result to
    the signed range of the given bit width, as the generated hardware would.
    """

    def clamp_result(self, result: torch.Tensor, quant_bits: int) -> torch.Tensor:
        """Clamp a tensor in place to the signed range of `quant_bits` bits.

        The range is `[-2**(quant_bits - 1), 2**(quant_bits - 1) - 1]`.

        Args:
            result: Tensor to clamp. It is modified in place.
            quant_bits: Bit width of the signed target range.

        Returns:
            The same tensor object as `result`, after clamping.
        """
        result.clamp_(-(2 ** (quant_bits - 1)), (2 ** (quant_bits - 1)) - 1)
        return result

    def intadd(
        self, a: torch.Tensor, b: torch.Tensor, c_quant_bits: int
    ) -> torch.Tensor:
        """Add two integer tensors and clamp the sum.

        Args:
            a: First operand, dtype `torch.int32`.
            b: Second operand, dtype `torch.int32`. Moved to the device of `a`.
            c_quant_bits: Bit width of the signed result range used for clamping.

        Returns:
            `a + b`, clamped to `c_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
        """
        check_dtype("a", a, torch.int32)
        check_dtype("b", b, torch.int32)
        c = a + b.to(a.device)

        return self.clamp_result(c, c_quant_bits)

    def intsub(
        self, a: torch.Tensor, b: torch.Tensor, c_quant_bits: int
    ) -> torch.Tensor:
        """Subtract two integer tensors and clamp the difference.

        Args:
            a: First operand, dtype `torch.int32`.
            b: Second operand, dtype `torch.int32`. Moved to the device of `a`.
            c_quant_bits: Bit width of the signed result range used for clamping.

        Returns:
            `a - b`, clamped to `c_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
        """
        check_dtype("a", a, torch.int32)
        check_dtype("b", b, torch.int32)

        c = a - b.to(a.device)
        return self.clamp_result(c, c_quant_bits)

    def intmatmul(
        self, a: torch.Tensor, b: torch.Tensor, c_quant_bits: int
    ) -> torch.Tensor:
        """Multiply two 2D integer matrices and clamp the product.

        The product is computed element by element on the CPU with Python loops,
        so it is slow for large matrices.

        Args:
            a: Left matrix of shape `(m, k)`, dtype `torch.int32`.
            b: Right matrix of shape `(k, n)`, dtype `torch.int32`.
            c_quant_bits: Bit width of the signed result range used for clamping.

        Returns:
            `a @ b` of shape `(m, n)` on the device of `a`, clamped to
            `c_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
        """
        check_dtype("a", a, torch.int32)
        check_dtype("b", b, torch.int32)

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
        x: torch.Tensor,
        w: torch.Tensor,
        x_quant_bits: int,
        w_quant_bits: int,
        b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute an integer multiply-accumulate `x @ w.T + b`.

        The product is clamped to `(x_quant_bits + 1) + (w_quant_bits + 1) + 1`
        bits. If a bias is given, it is added and the sum is clamped to one more
        bit.

        Args:
            x: Input tensor of shape `(..., in_features)`, dtype `torch.int32`.
            w: Weight matrix of shape `(out_features, in_features)`, dtype
                `torch.int32`.
            x_quant_bits: Bit width of the quantized inputs.
            w_quant_bits: Bit width of the quantized weights.
            b: Optional bias of shape `(out_features,)`, dtype `torch.int32`.

        Returns:
            The accumulated result of shape `(..., out_features)`.

        Raises:
            TypeError: If `x`, `w` or `b` is not a `torch.Tensor` or has the
                wrong dtype.
        """
        check_dtype("x", x, torch.int32)
        check_dtype("w", w, torch.int32)

        tmp = F.linear(x, w)

        tmp_quant_bits = (x_quant_bits + 1) + (w_quant_bits + 1)
        # TODO: should be dynamic, need to be solved in the future
        tmp_quant_bits = tmp_quant_bits + 1
        self.clamp_result(tmp, tmp_quant_bits)

        if b is not None:
            check_dtype("b", b, torch.int32)
            tmp = self.intadd(tmp, b, tmp_quant_bits + 1)

        return tmp

    def int_matmul_4d(
        self,
        inputs1: torch.Tensor,
        inputs2: torch.Tensor,
        operation_mode: str,
        outputs_quant_bits: int,
    ) -> torch.Tensor:
        """Compute an attention matrix product on 4D integer tensors.

        Args:
            inputs1: Left operand, dtype `torch.int32`.
            inputs2: Right operand, dtype `torch.int32`.
            operation_mode: `"score"` computes the attention scores `Q @ K^T`
                after swapping dimensions 1 and 2 of both inputs. `"att"`
                computes `softmax_out @ V` after swapping dimensions 1 and 2 of
                `inputs2`, then swaps dimensions 1 and 2 of the result.
            outputs_quant_bits: Bit width of the signed result range used for
                clamping.

        Returns:
            The product, clamped to `outputs_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
            NotImplementedError: If `operation_mode` is neither `"score"` nor
                `"att"`.
        """
        check_dtype("inputs1", inputs1, torch.int32)
        check_dtype("inputs2", inputs2, torch.int32)

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
        inputs1: torch.Tensor,
        inputs2: torch.Tensor,
        operation_mode: str,
    ) -> torch.Tensor:
        """Compute an attention matrix product on 4D float tensors.

        Floating-point counterpart of `int_matmul_4d`; the result is not clamped.

        Args:
            inputs1: Left operand, dtype `torch.float32`.
            inputs2: Right operand, dtype `torch.float32`.
            operation_mode: `"score"` computes `Q @ K^T`, `"att"` computes
                `softmax_out @ V`, with the same transpositions as `int_matmul_4d`.

        Returns:
            The product.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
            NotImplementedError: If `operation_mode` is neither `"score"` nor
                `"att"`.
        """
        check_dtype("inputs1", inputs1, torch.float32)
        check_dtype("inputs2", inputs2, torch.float32)

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
        self, a: torch.Tensor, b: torch.Tensor, c_quant_bits: int
    ) -> torch.Tensor:
        """Multiply two integer tensors element-wise and clamp the product.

        The product is computed on the CPU and moved back to the device of `a`.

        Args:
            a: First operand, dtype `torch.int32`.
            b: Second operand with the same shape as `a`, dtype `torch.int32`.
            c_quant_bits: Bit width of the signed result range used for clamping.

        Returns:
            `a * b`, clamped to `c_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
            AssertionError: If `a` and `b` have different shapes.
        """
        check_dtype("a", a, torch.int32)
        check_dtype("b", b, torch.int32)
        assert a.shape == b.shape

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        dot_product = a_cpu * b_cpu

        dot_product = dot_product.to(a.device)
        return self.clamp_result(dot_product, c_quant_bits)

    def int_dotproduct(
        self, a: torch.Tensor, b: torch.Tensor, c_quant_bits: int
    ) -> torch.Tensor:
        """Multiply two integer tensors element-wise and clamp the product.

        Despite its name, this does not sum the products. Unlike
        `inthadamardproduct`, the shapes only need to be broadcastable.

        Args:
            a: First operand, dtype `torch.int32`.
            b: Second operand, dtype `torch.int32`. Moved to the device of `a`.
            c_quant_bits: Bit width of the signed result range used for clamping.

        Returns:
            `a * b` on the device of `a`, clamped to `c_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
        """
        check_dtype("a", a, torch.int32)
        check_dtype("b", b, torch.int32)

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        dot_product = a_cpu * b_cpu

        dot_product = dot_product.to(a.device)
        return self.clamp_result(dot_product, c_quant_bits)

    def int_division(
        self, a: torch.Tensor, b: torch.Tensor, c_quant_bits: int
    ) -> torch.Tensor:
        """Divide two integer tensors with floor division and clamp the quotient.

        Both operands are cast to `torch.int32` on the CPU before dividing.

        Args:
            a: Dividend, dtype `torch.int32` or `torch.int64`.
            b: Divisor, dtype `torch.int32` or `torch.int64`.
            c_quant_bits: Bit width of the signed result range used for clamping.

        Returns:
            `a // b` on the device of `a`, clamped to `c_quant_bits` bits.

        Raises:
            TypeError: If an operand is not a `torch.Tensor` or has the wrong dtype.
        """
        check_dtype("a", a, torch.int32, torch.int64)
        check_dtype("b", b, torch.int32, torch.int64)

        a_cpu = a.cpu().to(torch.int32)
        b_cpu = b.cpu().to(torch.int32)

        division = a_cpu // b_cpu

        division = division.to(a.device)
        return self.clamp_result(division, c_quant_bits)
