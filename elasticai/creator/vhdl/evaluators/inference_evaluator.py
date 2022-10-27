import torch
from torch.utils.data import Dataset

from elasticai.creator.vhdl.evaluators.evaluator import Evaluator
from elasticai.creator.vhdl.quantized_modules.typing import QuantType


class FullPrecisisonInferenceEvaluator(Evaluator):
    def __init__(self, module: torch.nn.Module, data: Dataset) -> None:
        self.module = module
        self.data = data

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        predictions, labels = [], []
        for sample, label in self.data:
            prediction = self.module(sample)
            predictions.append(prediction)
            labels.append(label)
        return torch.tensor(predictions), torch.tensor(labels)


class QuantizedInferenceEvaluator(Evaluator):
    def __init__(
        self,
        module: torch.nn.Module,
        data: Dataset,
        input_quant: QuantType = lambda x: x,
        output_dequant: QuantType = lambda x: x,
    ) -> None:
        if not hasattr(module, "quantized_forward"):
            raise ValueError(
                "`module` must be an object that has a `qunatized_forward` function."
            )
        self.module = module
        self.data = data
        self.input_quant = input_quant
        self.output_dequant = output_dequant

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        predictions, labels = [], []
        for sample, label in self.data:
            quantized_data = self.input_quant(sample)
            prediction = self.module.quantized_forward(quantized_data)  # type: ignore
            dequantized_prediction = self.output_dequant(prediction)
            predictions.append(dequantized_prediction)
            labels.append(label)
        return torch.tensor(predictions), torch.tensor(labels)
