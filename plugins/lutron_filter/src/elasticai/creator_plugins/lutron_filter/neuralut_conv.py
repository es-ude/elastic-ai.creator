import torch as t
import torch.nn as tnn

import elasticai.creator.ir as ir
import elasticai.creator.ir2torch as ir2t
import elasticai.creator.torch2ir as t2ir

from .nn.binarize import Binarize


class NeuraLUTAssembleConv(tnn.Module):
    def __init__(
        self,
        kernel_size: int,
        in_channels: int,
        out_channels: int,
        width_factor: int,
        groups=1,
    ) -> None:

        self.dw_conv = tnn.Conv1d(
            in_channels=width_factor * in_channels,
            out_channels=width_factor * groups,
            kernel_size=kernel_size,
            groups=groups,
        )
        self.residual_conv = tnn.Conv1d(
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
        )
        tnn.init.kaiming_normal(self.dw_conv.weight, nonlinearity="relu")
        self.width_factor = width_factor
        self.inter_relu = tnn.ReLU()
        self.inter_bn = tnn.BatchNorm1d(groups * width_factor)
        self.pw_conv = tnn.Conv1d(
            in_channels=groups * width_factor,
            out_channels=out_channels,
            groups=groups,
            kernel_size=1,
        )
        self.final_bn = tnn.BatchNorm1d(out_channels)
        self.binarize = Binarize()

    def forward(self, x: t.Tensor) -> t.Tensor:
        residual = self.residual_conv(x)
        repetitions = [1 for _ in x.size()]
        repetitions[-2] = self.width_factor
        x = x.repeat(repetitions)
        x = self.dw_conv(x)
        x = self.inter_relu(x)
        x = self.pw_conv(x)
        x = residual + x
        return self.binarize(x)


def replace_conv_by_neuralutrons(
    root: ir2t.DataGraph, registry: ir.Registry[ir2t.DataGraph]
) -> tuple[ir2t.DataGraph, ir.Registry[ir2t.DataGraph]]:
    executor = ir.ExecutionOrderGraphReducer()

    return root, registry
