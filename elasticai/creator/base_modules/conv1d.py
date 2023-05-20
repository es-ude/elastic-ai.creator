from typing import Any

import torch

from elasticai.creator.base_modules.arithmetics import Arithmetics


class Conv1d(torch.nn.Conv1d):
    def __init__(
        self,
        arithmetics: Arithmetics,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: int | tuple[int] | str = 0,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        self._arithmetics = arithmetics

    @staticmethod
    def _flatten_tuple(x: int | tuple[int, ...]) -> int:
        return x[0] if isinstance(x, tuple) else x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # TODO: ADD PADDING !!!

        has_batches = inputs.dim() == 3
        if not has_batches:
            inputs = inputs.expand(1, -1, -1)

        batch_size, _, input_length = inputs.shape
        kernel_size = self._flatten_tuple(self.kernel_size)
        stride = self._flatten_tuple(self.stride)

        output_length = (input_length - kernel_size) // stride + 1
        outputs = torch.empty(batch_size, self.out_channels, output_length)

        weight = self._arithmetics.quantize(self.weight)
        bias = None if self.bias is None else self._arithmetics.quantize(self.bias)

        for window_start_idx in range(output_length):
            start_idx = window_start_idx * stride
            input_slice = inputs[:, :, start_idx : start_idx + kernel_size]

            weighted_slice = self._arithmetics.mul(input_slice, weight)
            output_single_conv = self._arithmetics.sum(weighted_slice, dim=(1, 2))

            if bias is not None:
                output_single_conv = self._arithmetics.add(output_single_conv, bias)

            outputs[:, :, window_start_idx] = output_single_conv.view(
                batch_size, self.out_channels
            )

        if not has_batches:
            outputs = outputs.squeeze(dim=0)

        return outputs
