import torch

from elasticai.creator.base_modules.conv1d import MathOperations as Conv1dOps
from elasticai.creator.base_modules.linear import MathOperations as LinearOps
from elasticai.creator.base_modules.lstm_cell import MathOperations as LSTMCellOps

from .arithmetic import CutToInteger, LinearArithmetic


class MathOperations(LinearOps, Conv1dOps, LSTMCellOps):
    """MathOperations for linear quantization.

    Attributes:
        config (LinearArithmetic): Configuration parameters for linear quantization.

    Returns:
        MathOperations: An instance of the MathOperations class.
    """

    config: LinearArithmetic

    def __init__(self, config: LinearArithmetic):
        self.config = config

        super().__init__()

    def quantize(self, a: torch.Tensor) -> torch.Tensor:
        """Quantizes the input tensor x.

        Args:
            x (torch.Tensor): The input tensor to be quantized.

        Returns:
            torch.Tensor: The quantized tensor.
        """
        return CutToInteger.apply(
            torch.clamp(
                input=a,
                min=self.config.minimum_as_rational,
                max=self.config.maximum_as_rational,
            ),
            self.config,
        )

    def add(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Performs element-wise addition between two tensors.

        Args:
            a (torch.Tensor): The first tensor.
            b (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The result of the element-wise addition.
        """
        return self.quantize(a=torch.add(input=a, other=b))

    def mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Performs element-wise multiplication between two tensors.

        Args:
            a (torch.Tensor): The first tensor.
            b (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The result of the element-wise multiplication.
        """
        return self.quantize(a=torch.mul(input=a, other=b))

    def matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Performs matrix multiplication between two tensors.

        Args:
            a (torch.Tensor): The first tensor.
            b (torch.Tensor): The second tensor.

        Returns:
            torch.Tensor: The result of the matrix multiplication.
        """
        return self.quantize(a=torch.matmul(input=a, other=b))
