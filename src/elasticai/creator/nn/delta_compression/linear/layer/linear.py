from typing import Any

from torch import Tensor
from torch.nn import Linear as TorchLinear

from elasticai.creator.arithmetic import (
    FxpArithmetic,
    FxpParams,
)
from elasticai.creator.nn.delta_compression import DeltaOperations, DeltaType
from elasticai.creator.nn.design_creator_module import DesignCreatorModule
from elasticai.creator.nn.fixed_point import MathOperations

type LinearDesign = (
    Any  # Placeholder for the actual design class that will be implemented later
)


class Linear(DesignCreatorModule, TorchLinear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        total_bits: int,
        frac_bits: int,
        delta_bits: int,
        delta_offset: int,
        delta_type: DeltaType,
        clamp: bool = False,
        bias: bool = True,
        device: Any = None,
    ) -> None:
        self._params = FxpParams(
            total_bits=total_bits, frac_bits=frac_bits, signed=True
        )
        self._config = FxpArithmetic(self._params)
        self._operations = MathOperations(config=self._config)
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
        )
        self._deltaops = DeltaOperations(
            fxp_arithmetic=self._config,
            delta_bits=delta_bits,
            delta_offset=delta_offset,
            delta_type=delta_type,
            clamp=clamp,
        )

    @property
    def _bias(self) -> Tensor:
        return Tensor([0] * self.out_features) if self.bias is None else self.bias

    def _ste(self, x: Tensor) -> Tensor:
        """Straight-through estimator: compress+inflate in forward, identity gradient."""
        q = self._operations.quantize(x)
        compressed = self._deltaops.inflate(self._deltaops.compress(q))
        return compressed.detach() + (q - q.detach())

    def forward(self, input: Tensor) -> Tensor:
        weight = self._ste(self.weight)
        if self.bias is None:
            return self._operations.matmul(input, weight.T)
        else:
            return self._operations.add(
                self._operations.matmul(input, weight.T),
                self._ste(self.bias),
            )

    def get_params(self) -> tuple[list[list[float]], list[float]]:
        bias = self._bias.tolist()
        weights = self.weight.tolist()
        return weights, bias

    def get_params_compressed(self) -> tuple[list[list[int]], list[int]]:
        c_weights = self._deltaops.compress(
            self._operations.quantize(self.weight)
        ).tolist()
        c_bias = self._deltaops.compress(self._operations.quantize(self._bias)).tolist()
        return c_weights, c_bias

    def create_design(self, name: str) -> LinearDesign:
        raise NotImplementedError("Design creation not implemented yet")
