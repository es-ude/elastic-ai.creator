import unittest

import torch
from torch.utils.data import Dataset, TensorDataset

from elasticai.creator.vhdl.evaluators.inference_evaluator import (
    FullPrecisisonInferenceEvaluator,
    QuantizedInferenceEvaluator,
)


class ModelMock(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1

    def quantized_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 2


def get_dataset(samples: list, labels: list) -> Dataset:
    return TensorDataset(torch.tensor(samples), torch.tensor(labels))


def to_list(x: torch.Tensor) -> list[float]:
    return x.detach().numpy().tolist()


class FullPrecisionInferenceEvaluatorTest(unittest.TestCase):
    def test_full_precision_inference_evaluator_empty_dataset(self) -> None:
        model = ModelMock()
        samples: list[float] = []
        labels: list[float] = []
        evaluator = FullPrecisisonInferenceEvaluator(
            module=model, data=get_dataset(samples, labels)
        )
        predictions, passed_through_labels = evaluator.run()
        self.assertEqual(labels, to_list(passed_through_labels))
        self.assertEqual(labels, to_list(predictions))

    def test_full_precision_inference_evaluator_filled_single_dim_dataset(self) -> None:
        model = ModelMock()
        samples: list[float] = [1, 2, 3]
        labels: list[float] = [2, 3, 4]
        evaluator = FullPrecisisonInferenceEvaluator(
            module=model, data=get_dataset(samples, labels)
        )
        predictions, passed_through_labels = evaluator.run()
        self.assertEqual(labels, to_list(passed_through_labels))
        self.assertEqual(labels, to_list(predictions))

    def test_full_precision_inference_evaluator_filled_multi_dim_dataset(self) -> None:
        model = ModelMock()
        samples: list[list[float]] = [[1, 2], [3, 4], [5, 6]]
        labels: list[list[float]] = [[2, 3], [4, 5], [6, 7]]
        evaluator = FullPrecisisonInferenceEvaluator(
            module=model, data=get_dataset(samples, labels)
        )
        predictions, passed_through_labels = evaluator.run()
        self.assertEqual(labels, to_list(passed_through_labels))
        self.assertEqual(labels, to_list(predictions))


class QuantizedInferenceEvaluatorTest(unittest.TestCase):
    def test_quantized_inference_evaluator_empty_dataset(self) -> None:
        model = ModelMock()
        samples: list[float] = []
        labels: list[float] = []
        evaluator = QuantizedInferenceEvaluator(
            module=model, data=get_dataset(samples, labels)
        )
        predictions, passed_through_labels = evaluator.run()
        self.assertEqual(labels, to_list(passed_through_labels))
        self.assertEqual(labels, to_list(predictions))

    def test_quantized_inference_evaluator_no_input_output_quantizers(self) -> None:
        model = ModelMock()
        samples: list[list[float]] = [[1, 2], [3, 4], [5, 6]]
        labels: list[list[float]] = [[3, 4], [5, 6], [7, 8]]
        evaluator = QuantizedInferenceEvaluator(
            module=model, data=get_dataset(samples, labels)
        )
        predictions, passed_through_labels = evaluator.run()
        self.assertEqual(labels, to_list(passed_through_labels))
        self.assertEqual(labels, to_list(predictions))

    def test_quantized_inference_evaluator_input_output_quantizers(self) -> None:
        model = ModelMock()
        samples: list[list[float]] = [[1, 2], [3, 4], [5, 6]]
        labels: list[list[float]] = [[2, 3], [4, 5], [6, 7]]
        evaluator = QuantizedInferenceEvaluator(
            module=model,
            data=get_dataset(samples, labels),
            input_quant=lambda x: x * 2,
            output_dequant=lambda x: x / 2,
        )
        predictions, passed_through_labels = evaluator.run()
        self.assertEqual(labels, to_list(passed_through_labels))
        self.assertEqual(labels, to_list(predictions))
