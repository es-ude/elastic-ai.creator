from typing import Callable

import torch
from torch.utils.data import Dataset

from elasticai.creator.vhdl.evaluators.evaluator import Evaluator
from elasticai.creator.vhdl.quantized_modules.typing import QuantType


def _run_inference_on_dataset(
    data: Dataset, predict_sample_fn: Callable[[torch.Tensor], torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    def to_tensor(x: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(x) if len(x) > 0 else torch.tensor([])

    predictions, labels = [], []
    for sample, label in data:
        prediction = predict_sample_fn(sample)
        predictions.append(prediction)
        labels.append(label)
    return to_tensor(predictions), to_tensor(labels)


class FullPrecisisonInferenceEvaluator(Evaluator):
    def __init__(self, module: torch.nn.Module, data: Dataset) -> None:
        self.module = module
        self.data = data

    def run(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _run_inference_on_dataset(self.data, self.module)


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
        def predict_sample(sample: torch.Tensor) -> torch.Tensor:
            quantized_data = self.input_quant(sample)
            prediction = self.module.quantized_forward(quantized_data)  # type: ignore
            return self.output_dequant(prediction)

        return _run_inference_on_dataset(self.data, predict_sample)
